import torch
import random

class Function:
    @staticmethod
    def get_function(name, x, params):
        if name == 'ackley':
            return Ackley(x, params)
        elif name == 'griewank':
            return Griewank(x, params)
        elif name == 'largeman':
            return Largeman(x, params)
        elif name == 'levy':
            return Levy(x, params)
        elif name == 'rastrigin':
            return Rastrigin(x, params)
        elif name == 'buckin':
            return Buckin(x, params)
        elif name == 'crossintray':
            return CrossInTray(x, params)
        elif name == 'dropwave':
            return DropWave(x, params)
        else:
            raise ValueError(f"Unknown function name: {name}")

#n dims - many local minima
class Ackley():
    def __init__(self, x, params):
        self.x = x
        self.params = params
        # self.dim = x.shape[1]

    def evaluate_function(self):
        a, b, c = self.params
        dim = self.x.shape[1]
        sum_sp = torch.sum(self.x**2, dim=1)  
        sum_cos = torch.sum(torch.cos(c * self.x), dim=1)  
        term1 = -a * torch.exp(-b * torch.sqrt(sum_sp / dim))
        term2 = -torch.exp(sum_cos / dim)
        result = term1 + term2 + a + torch.exp(torch.tensor(1.0))
        return result.unsqueeze(1)

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

class Griewank():
    def __init__(self, x, params):
        self.x = x
        self.params = params

    def evaluate_function(self):
        a, b = self.params
        d = self.x.shape[-1]
        term1 = torch.sum(self.x**2, dim=1) / 4000  
        indices = torch.arange(1, d + 1, dtype=torch.float32).unsqueeze(0)  
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

class Largeman():
    def __init__(self, x, params):
        self.x = x
        self.params = params

    def evaluate_function(self):
        m, c, A = self.params
        x_expanded = self.x.unsqueeze(1)  
        A_expanded = A.unsqueeze(0)

        sum_term = torch.sum((x_expanded - A_expanded) ** 2, dim=-1)  
        exp_term = torch.exp(-sum_term / torch.pi)
        cos_term = torch.cos(torch.pi * sum_term)

        result = torch.sum(c * exp_term * cos_term, dim=-1)  
        return result.unsqueeze(1) 

    @staticmethod
    def generate_functions(num, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num):
                m = random.randint(3, 10)
                c = torch.tensor([random.uniform(0.5, 5.0) for _ in range(m)], dtype=torch.float32)
                A = torch.tensor([[random.uniform(0, 10) for _ in range(dim)] for _ in range(m)], dtype=torch.float32)
                functions[dim].append((m, c, A))
        return functions

class Levy():
    def __init__(self, x, params):
        self.x = x
        self.params = params

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
class Rastrigin():
    def __init__(self, x, params):
        self.x = x
        self.params = params

    def evaluate_function(self):
        a, b = self.params
        d = self.x.shape[-1]
        sum_term = torch.sum(self.x**2 - a * torch.cos(b * self.x), dim=-1)
        return (a * d + sum_term).unsqueeze(1)

    @staticmethod
    def generate_functions(num_functions, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num_functions):
                a = random.uniform(5, 15)
                b = random.uniform(1.5 * torch.pi, 2.5 * torch.pi)
                functions[dim].append([a, b])
        return functions

class Buckin():
    def __init__(self, x, params):
        self.params = params
        self.x = x

    def evaluate_function(self):
        a, b, c = self.params
        x1 = self.x[:, 0]
        x2 = self.x[:, 1]
        return (a * torch.sqrt(torch.abs(x2 - c * x1**2)) + c * torch.abs(x1 + b)).unsqueeze(1)

    @staticmethod
    def generate_functions(num_function, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num_function):
                a = random.uniform(50, 200)
                b = random.uniform(5, 20)
                c = random.uniform(0, 1)
                functions[dim].append([a, b, c])
        return functions
class CrossInTray():
    def __init__(self, x, params):
        self.params = params
        self.x = x

    def evaluate_function(self):
        a, b, c = self.params
        x1 = self.x[:, 0]
        x2 = self.x[:, 1]
        return (a * torch.sin(x1) * torch.sin(x2) * torch.exp(torch.abs(b - torch.sqrt(x1**2 + x2**2) / torch.pi)) ** c).unsqueeze(1)

    @staticmethod
    def generate_functions(num_function, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num_function):
                a = random.uniform(-1, 0)
                b = random.uniform(50, 200)
                c = random.uniform(0, 0.5)
                functions[dim].append([a, b, c])
        return functions

class DropWave():
    def __init__(self, x, params):
        self.params = params
        self.x = x

    def evaluate_function(self):
        a, b, c, d = self.params
        x1 = self.x[:, 0]
        x2 = self.x[:, 1]
        sum_sq = x1**2 + x2**2
        return (-(a + torch.cos(b * torch.sqrt(sum_sq))) / (c * sum_sq + d)).unsqueeze(1)

    @staticmethod
    def generate_functions(num_function, dims):
        functions = {}
        for dim in dims:
            functions[dim] = []
            for _ in range(num_function):
                a = random.uniform(-5, 5)
                b = random.uniform(8, 20)
                c = random.uniform(0.1, 0.9)
                d = random.uniform(-5, 5)
                functions[dim].append([a, b, c, d])
        return functions
