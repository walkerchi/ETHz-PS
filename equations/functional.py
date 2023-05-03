import torch 
import numpy as np
pi = np.pi
def gradient(y, x):
    """
        \sum_{y_index}  partial y[y_index] / partial x
        Parameters:
        -----------
            y: torch.Tensor([...y_, y_dim])
            x: torch.Tensor([...x_, x_dim])
        Returns:
        --------
            torch.Tensor([...y_, x_dim])
    """
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
def partial_derivative(y, x, y_index=None, x_index=slice(None)):
    """
        parital y[y_index] / partial x[x_index]
        if do not pass the y_index and x_index, 
            it will do the same as the gradient function
        Parameters:
        -----------
            y: torch.Tensor([...y_, y_dim])
            x: torch.Tensor([...x_, x_dim])
            y_index: int | None if None sum all the gradient from y
            x_index: int | slice | None if None doo
        Returns:
        --------
            torch.Tensor([...x_, 1])
    """
    if y_index is None:
        grad = torch.ones_like(y)
    else:
        grad = torch.zeros_like(y)
        grad[...,y_index] = 1
    grad_x = torch.autograd.grad(y, x, grad, create_graph=True)[0]
    return grad_x[x_index]

def torch_uniform(a, b, size):
    return (b - a) * torch.rand(size) + a
def torch_normal(mu, sigma, size):
    if sigma == 0:
        return torch.full(size, mu)
    return sigma * torch.randn(size) + mu
def mse(x):
    return (x**2).mean()