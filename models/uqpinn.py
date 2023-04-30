import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from itertools import chain
from tqdm import tqdm

if __name__ == '__main__':
    from utils import MLP
else:
    from .utils import MLP

EPS = 1e-8
np.random.seed(1234)
torch.manual_seed(1234)

class UQPINN(nn.Module):
    """
        Uncertainty Quantification Physics Informed Neural Network
        t = T(x, y)
        z = Q(x, y)
        y = P(x, z)
        Discriminator: P(t|x, y) -> 1 P(t|x, z) -> 0
        Generator: P(t|x, z) -> 1  |pde(x, z) - f| -> 0
    """
    def __init__(self, x_dim=1, y_dim=1, z_dim=1,
                 n_layer_p=4, n_layer_q=4, n_layer_t=2,
                 n_hidden_p=50, n_hidden_q=50, n_hidden_t=50, 
                 lambd=1.5, beta=1.0):
        super().__init__()

        self.lambd = lambd
        self.beta = beta
        self.z_dim = z_dim
        self.P = MLP(x_dim + z_dim, y_dim, n_hidden_p, n_layer_p)
        self.Q = MLP(x_dim + y_dim, z_dim, n_hidden_q, n_layer_q)
        self.T = MLP(x_dim + y_dim, 1, n_hidden_t, n_layer_t)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def generator_parameters(self):
        return chain(self.P.parameters(), self.Q.parameters())

    def discriminator_parameters(self):
        return self.T.parameters()

    def generator_loss(self, equation, z_f, z_u):
        x_u, x_f = equation.x_u_norm, equation.x_f_norm
        y_u_pred = self.P(torch.cat([x_u, z_u], dim=1))
        y_f_pred = self.P(torch.cat([x_f, z_f], dim=1))
        r_f_pred = equation.pde(y_f_pred, x_f,use_jacobian=True) - equation.f(x_f)

        z_u_pred = self.Q(torch.cat([x_u, y_u_pred], dim=1))
        # z_f_pred = self.Q(torch.cat([x_f, r_f_pred], dim=1))
        T_u_pred = self.T(torch.cat([x_u, y_u_pred], dim=1)) 

        loss = T_u_pred.mean() - (1 - self.lambd) * ((z_u - z_u_pred)**2).mean() + self.beta * (r_f_pred**2).mean()

        return loss

    def discriminator_loss(self, equation, z_f, z_u):
        x_u, y_u = equation.x_u_norm, equation.y_u
        y_u_pred = self.P(torch.cat([x_u, z_u], dim=1))
        t_u_real = torch.sigmoid(self.T(torch.cat([x_u, y_u], dim=1)))
        t_u_pred = torch.sigmoid(self.T(torch.cat([x_u, y_u_pred], dim=1)))

        loss = - (torch.log(1 - t_u_real + EPS) + torch.log(t_u_pred + EPS)).mean()

        return loss
    
    def fit(self, equation, epoch=2000, k1=1, k2=5, print_every_epoch=100, lr=1e-4):
        optimizer_G = torch.optim.Adam(self.generator_parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(self.discriminator_parameters(), lr=lr)

        pbar = tqdm(total=epoch, unit="epoch")
        self.train()
        losses = {
            "loss_G": [],
            "loss_D": []
        }
        for ep in range(epoch):
            z_u = torch.randn(equation.N_u, self.z_dim).to(self.device)
            z_f = torch.randn(equation.N_f, self.z_dim).to(self.device)

            for _ in range(k1):
                optimizer_D.zero_grad()
                loss = self.discriminator_loss(equation, z_f, z_u)
                loss.backward()
                optimizer_D.step()

            for _ in range(k2):
                optimizer_G.zero_grad()
                loss = self.generator_loss(equation, z_f, z_u)
                loss.backward()
                optimizer_G.step()

            pbar.update(1)
            if (ep+1)%print_every_epoch == 0:
                self.eval()
                loss_D = self.discriminator_loss(equation, z_f, z_u)
                loss_G = self.generator_loss(equation, z_f, z_u)
                losses["loss_G"].append(loss_G.item())
                losses["loss_D"].append(loss_D.item())
                pbar.set_postfix({'loss_G': loss_G.item(), 'loss_D': loss_D.item()})
                self.train()

        losses["loss_G"] = np.array(losses["loss_G"])
        losses["loss_D"] = np.array(losses["loss_D"])
        return losses

    def predict(self, equation, x, n_samples=2000, batch_size=4):
        assert n_samples % batch_size == 0
        self.eval()
        y = torch.zeros(n_samples, len(x), equation.y_dim).to(self.device)
        z = torch.zeros(batch_size * len(x), self.z_dim).to(self.device)
        x = equation.norm_x(x)
        x = x.repeat(batch_size, 1)
        with torch.no_grad():
            pbar = tqdm(total=n_samples, unit=f"sample")
            for i in range(0, n_samples, batch_size):
                torch.nn.init.normal_(z, mean=0, std=1)
                y[i:i+batch_size] = self.P(torch.cat([x, z], dim=-1)).reshape([batch_size, -1, equation.y_dim])
                pbar.update(batch_size)
        mu, std = y.mean(dim=0), y.std(dim=0)
        return mu, std