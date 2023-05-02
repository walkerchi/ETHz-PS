import torch 
import torch.nn as nn
import torch.nn.functional as F 


class MLP(nn.Module):
    def __init__(self, 
        input_dim,
        output_dim,
        hidden_dim,
        num_hidden,
        activation=torch.tanh
        ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
        ])
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, x):
        x = x.type(self.dtype)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    

class StackMLP(nn.Module):
    def __init__(self, 
        input_dim,
        output_dim,
        hidden_dim,
        num_hidden,
        activation=torch.tanh
        ):
        super().__init__()
        self.mlps = nn.ModuleList([
            MLP(input_dim, 1, hidden_dim, num_hidden, activation),
        ])
        for _ in range(output_dim-1):
            self.mlps.append(MLP(1, 1, hidden_dim, num_hidden, activation))

    def forward(self, x):
        y = []
        for mlp in self.mlps:
            x = mlp(x)
            y.append(x)
        return torch.cat(y, dim=1)
    
    @property
    def stage(self):
        return self.mlps