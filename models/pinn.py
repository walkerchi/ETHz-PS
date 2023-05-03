import torch 
import torch.nn as nn 
import numpy as np
from tqdm import tqdm 
from .utils import StackMLP, MLP


class PINN(nn.Module):
    def __init__(self, nn, n_hidden, n_layer, x_dim, y_dim, **kwargs):
        super().__init__()
        self.nn = nn(x_dim, y_dim, n_hidden, n_layer)
    @property
    def device(self):
        return next(self.parameters()).device
    def loss(self, equation):
        x_u, y_u = equation.x_u_norm, equation.y_u
        x_f      = equation.x_f_norm
        y_u_pred = self.nn(x_u)
        y_f_pred = self.nn(x_f)
        if hasattr(equation, "corrent_y"):
            y_u_pred = equation.correct_y(y_u_pred)
            y_f_pred = equation.correct_y(y_f_pred)
        regression_loss = torch.mean((y_u_pred - y_u)**2)
        pde_loss = equation.pde_loss(y_f_pred)
        return regression_loss + pde_loss

    def fit(self, equation, epoch=2000, print_every_epoch=100, lr=1e-4, **kwargs):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        pbar = tqdm(total=epoch, unit="epoch")
        losses = {"loss":[]}
        for ep in range(epoch):
            optimizer.zero_grad()
            loss = self.loss(equation)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            if (ep+1)%print_every_epoch == 0:
                self.eval()
                loss = self.loss(equation)
                losses["loss"].append(loss.item())
                pbar.set_postfix({'loss': loss.item()})
                self.train()
        return losses
    
    def predict(self, equation, **kwargs):
        """
            Parameters:
            -----------
                equation: Equation
                    The equation to solve
            Returns:
            --------
                y: torch.Tensor [N_ref, y_dim]
                    The predicted solution
        """
        self.eval()
        x = equation.norm_x(equation.x_ref)
        with torch.no_grad():
            y = self.nn(x)
        return y