
import torch
import random

class Function:
    @staticmethod
    def get_function(name, x, params, device):
        functions = {
            'ackley': Ackley,
            'griewank': Griewank,
            'largeman': Largeman,
            'levy': Levy,
            'styblinski_tang': StyblinskiTang,
            'zakharov': Zakharov,
            'schwefel': Schwefel
        }
        if name in functions:
            return functions[name](x, params, device)
        else:
            raise ValueError(f"Unknown function name: {name}")

class Ackley:
    def __init__(self, x, params, device):
        self.device = device
        self.x = x.to(self.device)
        self.params = [p.to(self.device) if isinstance(p, torch.Tensor) else torch.tensor(p).to(self.device) for p in params]

    def evaluate_function(self):
        a, b, c = self.params
        dim = self.x.shape[1]
        sum_sp = torch.sum(self.x**2, dim=1)
        sum_cos = torch.sum(torch.cos(c * self.x), dim=1)
        term1 = -a * torch.exp(-b * torch.sqrt(sum_sp / dim))
        term2 = -torch.exp(sum_cos / dim)
        result = term1 + term2 + a + torch.exp(torch.tensor(1.0).to(self.device))
        return torch.abs(result).unsqueeze(1)

    @staticmethod
    def generate_functions(num_functions, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num_functions):
                a = random.uniform(15, 25)
                b = random.uniform(0.1, 0.3)
                c = random.uniform(1.5 * torch.pi, 2.5 * torch.pi)
                functions[dim].append([a, b, c])
        return functions

class Griewank:
    def __init__(self, x, params, device):
        self.device = device
        self.x = x.to(self.device)
        self.params = [p.to(self.device) if isinstance(p, torch.Tensor) else torch.tensor(p).to(self.device) for p in params]

    def evaluate_function(self):
        a, b = self.params
        d = self.x.shape[-1]
        term1 = torch.sum(self.x**2, dim=1) / 4000
        indices = torch.arange(1, d + 1, dtype=torch.float32).unsqueeze(0).to(self.device)
        term2 = torch.prod(torch.cos(self.x / torch.sqrt(indices)), dim=1)
        return (term1 - term2 + 1).unsqueeze(1)

    @staticmethod
    def generate_functions(num, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num):
                a = random.uniform(1/5000, 1/3000)
                b = random.uniform(0.5, 1.5)
                functions[dim].append([a, b])
        return functions

class Largeman:
    def __init__(self, x, params, device):
        self.device = device
        self.x = x.to(self.device)
        self.params = [
            p.to(self.device) if isinstance(p, torch.Tensor) else torch.tensor(p).to(self.device)
            for p in params
        ]

    def evaluate_function(self):
        m, c, A = self.params
        d = self.x.shape[1]
        results = torch.zeros(self.x.shape[0], device=self.device)
        for i in range(m):
            diff = self.x - A[i]
            sum_sq = torch.sum(diff**2, dim=1)
            term = c[i] * torch.exp(-1 / torch.pi * sum_sq) * torch.cos(torch.pi * sum_sq)
            results += term
        return torch.abs(results).unsqueeze(1)

    @staticmethod
    def generate_functions(num, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num):
                m = random.randint(3, 7)
                c = [random.uniform(1, 5) for _ in range(m)]
                A = torch.rand((m, dim)) * 10  # Random values in [0, 10]
                functions[dim].append([m, c, A])
        return functions

class Levy:
    def __init__(self, x, params, device):
        self.device = device
        self.x = x.to(self.device)
        self.params = [p.to(self.device) if isinstance(p, torch.Tensor) else torch.tensor(p).to(self.device) for p in params]

    def evaluate_function(self):
        a, b, c = self.params
        w = 1 + (self.x - 1) / 4

        term1 = a * torch.sin(torch.pi * w[:, 0]) ** 2
        term2 = torch.sum((w[:, :-1] - 1) ** 2 * (1 + b * torch.sin(torch.pi * w[:, :-1] + 1) ** 2), dim=1)
        term3 = (w[:, -1] - 1) ** 2 * (1 + c * torch.sin(2 * torch.pi * w[:, -1]) ** 2)

        return (term1 + term2 + term3).unsqueeze(1)

    @staticmethod
    def generate_functions(num, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num):
                a = random.uniform(0.5, 2.0)
                b = random.uniform(5, 20)
                c = random.uniform(0.5, 2.0)
                functions[dim].append([a, b, c])
        return functions


class Schwefel:
    def __init__(self, x, params, device):
        self.device = device
        self.x = x.to(self.device)
        self.params = [torch.tensor(p).to(self.device) for p in params]

    def evaluate_function(self):
        d = self.x.shape[-1]
        a, b = self.params
        term1 = a * d
        indices = torch.arange(1, d + 1, dtype=torch.float32).unsqueeze(0).to(self.device)
        term2 = torch.sum(self.x * torch.sin(torch.sqrt(torch.abs(b*self.x))), dim=1)
        return (term1 - term2).unsqueeze(1)

    @staticmethod
    def generate_functions(num_functions, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num_functions):
                a = random.uniform(300, 500)
                b = random.uniform(1.5 * torch.pi, 2.5 * torch.pi)
                functions[dim].append([a, b])
        return functions

class StyblinskiTang:
    def __init__(self, x, params, device):
        self.device = device
        self.x = x.to(self.device)
        self.params = [torch.tensor(p).to(self.device) for p in params]

    def evaluate_function(self):
        d = self.x.shape[1]
        a, b, c = self.params
        result = a * torch.sum(self.x**4 - b * self.x**2 + c * self.x, dim=1)
        return torch.abs(result).unsqueeze(1)

    @staticmethod
    def generate_functions(num_functions, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num_functions):
                a = random.uniform(0.1, 0.9)
                b = random.uniform(10, 18)
                c = random.uniform(1, 7)
                functions[dim].append([a, b, c])
        return functions

class Zakharov:
    def __init__(self, x, params, device):
        self.device = device
        self.x = x.to(self.device)
        self.params = [torch.tensor(p).to(self.device) for p in params]

    def evaluate_function(self):
        d = self.x.shape[1]
        a, b, c = self.params
        term1 = torch.sum(self.x**2, dim=1)
        term2 = b*torch.sum(a * torch.arange(1, d + 1, device=self.device) * self.x, dim=1) ** 2
        term3 = c*torch.sum(a * torch.arange(1, d + 1, device=self.device) * self.x, dim=1) ** 4
        result = term1 + term2 + term3
        return torch.abs(result).unsqueeze(1)

    @staticmethod
    def generate_functions(num_functions, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num_functions):
                a = random.uniform(0.1, 0.8)
                b = random.uniform(0.01, 2)
                c = random.uniform(0.01, 2)
                functions[dim].append([a, b, c]) 
        return functions
