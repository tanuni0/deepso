import torch
import torch_geometric
from torch_geometric.data import Data
from net import PSOGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class OriPSO():
    def __init__(self, X, func, W, C1, C2, lower_bound, upper_bound):
        super().__init__()
        self.device = device
        self.X = X.to(self.device)
        self.func = func
        
        # Convert W, C1, and C2 to tensors if they are not
        self.W = torch.tensor(W, device=self.device) if not torch.is_tensor(W) else W.to(self.device)
        self.C1 = torch.tensor(C1, device=self.device) if not torch.is_tensor(C1) else C1.to(self.device)
        self.C2 = torch.tensor(C2, device=self.device) if not torch.is_tensor(C2) else C2.to(self.device)

        self.lower_bound = torch.tensor(lower_bound, device=self.device) if not torch.is_tensor(lower_bound) else lower_bound.to(self.device)
        self.upper_bound = torch.tensor(upper_bound, device=self.device) if not torch.is_tensor(upper_bound) else upper_bound.to(self.device)
        
        self.num_particle, self.dim = X.shape
        self.V = torch.zeros((self.num_particle, self.dim)).to(self.device)
        self.P = X.detach().to(self.device)
        self.P_best = torch.full((self.num_particle, 1), float('inf')).to(self.device)
        self.G = None
        self.global_best = torch.tensor(float('inf'), device=self.device)
        self.global_best_history = []

    def initial_global_best(self):
      fitnesses = self.func(self.X)
      min_fitness, min_id = torch.min(fitnesses, dim=0)
      self.global_best = min_fitness
      self.G = self.X.detach()[min_id].to(self.device) 
      return self.G

    def update_position(self):
      new_velocity = self.W.reshape(-1, 1) * self.V + torch.rand(1, device=self.device) * self.C1.reshape(-1, 1) * (self.P - self.X) + torch.rand(1, device=self.device) * self.C2.reshape(-1, 1) * (self.G - self.X)
      new_position = self.X + new_velocity
      new_position = torch.clamp(new_position, self.lower_bound, self.upper_bound)

      self.V = new_velocity.detach().to(self.device)
      self.X = new_position.detach().to(self.device)
      return new_position

    def update_fitness(self, new_position):
      fitnesses = self.func(new_position).reshape(-1, 1)
      fitnesses_no_grad = fitnesses.detach().to(self.device)

      improve = fitnesses_no_grad < self.P_best
      self.P_best = torch.where(improve, fitnesses_no_grad, self.P_best)
      self.P = torch.where(improve.reshape(-1, 1), new_position.detach().to(self.device), self.P)

      min_fitness, min_id = torch.min(fitnesses_no_grad, dim=0)
      if self.global_best > min_fitness:
          self.global_best = min_fitness
          self.G = new_position.detach()[min_id].to(self.device)

      self.global_best_history.append(self.global_best)
      return torch.mean(fitnesses)

    def run(self):
      if self.G is None:
          self.G = self.initial_global_best()
      new_position = self.update_position()
      mean_fitness = self.update_fitness(new_position)
      return self.G, self.global_best, mean_fitness

    
