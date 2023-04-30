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

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)