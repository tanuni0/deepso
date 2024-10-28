import torch
import torch_geometric
from torch_geometric.data import Data
from net import PSOGNN

class PSO():
    def __init__(self, X, func, model, lower_bound, upper_bound, device='cuda', patience=5):
        super().__init__()
        self.device = device  
        self.X = X.to(self.device)
        self.func = func
        self.lower_bound = torch.tensor(lower_bound, device=self.device) if not torch.is_tensor(lower_bound) else lower_bound.to(self.device)
        self.upper_bound = torch.tensor(upper_bound, device=self.device) if not torch.is_tensor(upper_bound) else upper_bound.to(self.device)
        self.num_particle, self.dim = X.shape
        self.V = torch.zeros((self.num_particle, self.dim)).to(self.device)
        self.P = X.detach().to(self.device)  
        self.P_best = torch.full((self.num_particle, 1), float('inf')).to(self.device)  
        self.G = None  
        self.global_best = torch.tensor(float('inf'), device=self.device)  
        self.global_best_history = []
        self.model = model.to(self.device)
        self.patience = patience  # Số vòng lặp không cải thiện trước khi dừng

    def initial_global_best(self):
        fitnesses = self.func(self.X)
        min_fitness, min_id = torch.min(fitnesses, dim=0)
        self.global_best = min_fitness
        self.G = self.X.detach()[min_id].to(self.device)  
        return self.G

    def update_position(self, W, C1, C2):
        new_velocity = W.reshape(-1, 1) * self.V + torch.rand(1, device=self.device) * C1.reshape(-1, 1) * (self.P - self.X) + torch.rand(1, device=self.device) * C2.reshape(-1, 1) * (self.G - self.X)
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

        no_improvement_counter = 0  
        best_loss = self.global_best.item() if self.global_best is not None else None

        while no_improvement_counter < self.patience:
            X_dim = self.X.shape[1]
            Y_dim = self.model.node_input_dim
            if X_dim < Y_dim:
                padding_needed = Y_dim - X_dim
                X_padding = torch.nn.functional.pad(self.X, (0, padding_needed), "constant", 0)
            else:
                X_padding = self.X

            edge_index = torch.combinations(torch.arange(self.num_particle, device=self.device), r=2).t().contiguous().to(self.device)
            data = Data(x=X_padding, edge_index=edge_index)
            weight = self.model(data)
            W, C1, C2 = weight[:, 0], weight[:, 1], weight[:, 2]
            new_position = self.update_position(W, C1, C2)
            mean_fitness = self.update_fitness(new_position)

            current_loss = mean_fitness.item() if mean_fitness is not None else None

            if best_loss is None or current_loss is None:
                best_loss = current_loss
                no_improvement_counter = 0
            else:
                if best_loss - current_loss > 1e-4:
                    best_loss = current_loss
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

        return self.G, self.global_best, mean_fitness
