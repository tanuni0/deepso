import torch
import torch_geometric
from torch_geometric.data import Data
from net import PSOGNN

class OriPSO():
    def __init__(self, pop_size, func, dim, lower_bound, upper_bound, W, C1, C2, X, device, max_iter=100, tolerance=1e-6, patience=10):
        super().__init__()
        self.device = device
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = torch.tensor(lower_bound, device=device)
        self.upper_bound = torch.tensor(upper_bound, device=device)
        self.W = torch.tensor(W, device=device)
        self.C1 = torch.tensor(C1, device=device)
        self.C2 = torch.tensor(C2, device=device)
        self.X = X.to(device)
        self.V = torch.zeros((pop_size, dim), device=device)
        self.P = X.detach().clone().to(device)                                         
        self.P_best = torch.full((pop_size, 1), float('inf'), device=device)     
        self.G = None                                               
        self.global_best = torch.tensor(float('inf'), device=device)                        
        self.global_best_history = []
        self.max_iter = max_iter  
        self.tolerance = tolerance  
        self.patience = patience  
        self.stopping_iter = 0  

    def initial_global_best(self):
        fitnesses = self.func(self.X.to(self.device))
        min_fitness, min_id = torch.min(fitnesses, dim=0)
        min_id = min_id.to(self.device)  
        G = self.X.detach()[min_id].clone().to(self.device)
        self.global_best = min_fitness.to(self.device)
        return G

    def update_position_all_swarm(self, G):
        new_velocity = (
            self.W.reshape(-1,1) * self.V + 
            torch.rand(1, device=self.device) * self.C1.reshape(-1,1) * (self.P - self.X) + 
            torch.rand(1, device=self.device) * self.C2.reshape(-1,1) * (G - self.X)
        )
        new_position = self.X + new_velocity
        new_position = torch.clamp(new_position, self.lower_bound, self.upper_bound)
        self.V = new_velocity.detach().to(self.device)
        self.X = new_position.to(self.device)
        return self.X

    def update_fitness(self, new_position):
        fitnesses = self.func(new_position.to(self.device)).reshape(-1, 1)
        fitnesses_no_grad = fitnesses.detach().to(self.device)
        improve = fitnesses_no_grad < self.P_best.to(self.device)
        self.P_best = torch.where(improve, fitnesses_no_grad, self.P_best.to(self.device))
        self.P = torch.where(improve.reshape(-1, 1), new_position.to(self.device), self.P.to(self.device))
        min_fitness, min_id = torch.min(fitnesses, dim=0)
        if self.global_best > min_fitness:
            self.global_best = min_fitness.to(self.device)
            self.G = new_position.detach()[min_id].clone().to(self.device)
        self.global_best_history.append(self.global_best)
        return torch.mean(fitnesses)

    def run(self):
        if self.G is None:
            self.G = self.initial_global_best()

        previous_best = self.global_best
        for iter in range(self.max_iter):
            new_position = self.update_position_all_swarm(self.G)
            mean_fitness = self.update_fitness(new_position)
            
            improvement = previous_best - self.global_best

            if improvement < self.tolerance:  
                self.stopping_iter += 1
            else:
                self.stopping_iter = 0  
                previous_best = self.global_best

            if self.stopping_iter >= self.patience:
                break

        return self.G, self.global_best, mean_fitness
