import torch
import torch_geometric
from torch_geometric.data import Data
from net import PSOGNN

class PSO():
  def __init__(self, X, func, model, lower_bound, upper_bound):
    super().__init__()
    self.X = X
    self.func = func
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.num_particle, self.dim = X.shape
    self.V = torch.zeros((self.num_particle, self.dim))
    self.P = X.detach()                                                    # position of all particles for p_best
    self.P_best = torch.full((self.num_particle, 1), float('inf'))         # best fitness of each particle
    self.G = None                                                          # position of a particle that has g_best
    self.global_best = torch.inf                                           # best fitness value of all particles
    self.global_best_history = []
    self.model = model

  def initial_global_best(self):
      fitnesses = self.func(self.X)
      min_fitness, min_id = torch.min(fitnesses, dim = 0)
      self.global_best = min_fitness
      return self.X.detach()[min_id]

  def update_position(self, W, C1, C2):
      new_velocity = W.reshape(-1,1) * self.V + torch.rand(1) * C1.reshape(-1,1) * (self.P - self.X) + torch.rand(1) * C2.reshape(-1,1) * (self.G - self.X)
      new_position = self.X + new_velocity
      new_position = torch.clamp(new_position, self.lower_bound, self.upper_bound)
      
      self.V = new_velocity.detach()
      self.X = new_position.detach()
      return new_position
    
  def update_fitness(self, new_position):
      fitnesses = self.func(new_position).reshape(-1,1)
      fitnesses_no_grad = fitnesses.detach()
      # print('fitnesses_no_grad:', fitnesses_no_grad.shape)
      # print('P_best:', self.P_best.shape)  
      
      improve = fitnesses_no_grad < self.P_best
      self.P_best = torch.where(improve, fitnesses_no_grad, self.P_best)
      self.P = torch.where(improve.reshape(-1,1), new_position.detach(), self.P)
      
      min_fitness, min_id = torch.min(fitnesses_no_grad, dim = 0)
      if self.global_best > min_fitness:
        self.global_best = min_fitness
        self.G = new_position.detach()[min_id]

      self.global_best_history.append(self.global_best)
      return torch.mean(fitnesses)


  def run(self):
      if self.G is None:
        self.G = self.initial_global_best()
      
      edge_index = torch.combinations(torch.arange(self.num_particle), r=2).t().contiguous()
      data = Data(x = self.X, edge_index= edge_index)
      # model = PSOGNN(node_input_dim=self.dim)
      
      weight = self.model(data)
      W, C1, C2 = weight[:,0], weight[:, 1], weight[:,2]
      
      new_position = self.update_position(W, C1, C2)
      mean_fitness = self.update_fitness(new_position)
      return self.G, self.global_best, mean_fitness
    
