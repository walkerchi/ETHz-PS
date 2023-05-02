import torch 
import scipy.io
import numpy as np
from ..base import Equation
from ..functional import gradient, pi, torch_uniform, torch_normal, mse
class Burgers(Equation):
    """
    Burgers
        x and t are all considered as dimension in x variable
        ut + u ux - nu uxx = 0      x in [-1, 1], t in [0, 1]
        u(0, x) = -sin(pi * x)
        u(t, -1) = u(t, 1) = 0

        nu = 0.01 / np.pi
    """
    x_dim = 2  #[t, x]
    y_dim = 1
    x_names = ['t','x']
    y_names = ['u']
    x_boundary = [[0, 1], [-1, 1]]
    has_exact_solution = False

    nu = 0.01 / np.pi

    def generate_boundary_data(self, n):
        """
            Generate boundary data 
                u(0, x) = -sin(pi * (x + 2e)) + e        e ~ normal(0, 1/exp(3|x|)/noise) 
                u(t, -1) = u(t, 1) = 0
                x in [-1, 1], t in [0, 1]
            Parameters:
            -----------
                n: number of data point
                
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([n, y_dim])
        """
        assert n % 3 == 0
        t = torch.zeros([n // 3])
        x = torch_uniform(-1, 1, [n // 3])
        if self.noise == 0:
            e = torch.zeros([n // 3])
        else:
            e = torch_normal(0, torch.exp(3 * torch.abs(x))/self.noise, [n // 3])
        # e  = torch.randn([n // 3, 1]) / torch.exp(3 * torch.abs(x)) / self.noise
        x1 = torch.stack([t, x], dim=-1)
        y1 = - torch.sin(pi * (x + 2 * e)) + e
        y1 = y1[:, None]
        
        t = torch_uniform(0, 1, [2 * n // 3])
        x = torch.tensor([-1.0, 1.0]).repeat(n//3)
        x2 = torch.stack([t, x], dim=-1)
        y2 = torch.zeros([2 * n // 3, 1])

        x = torch.cat([x1, x2], dim=0)
        y = torch.cat([y1, y2], dim=0)

        return x , y 
    
    def generate_collosion_data(self, n):
        """
            Generate collosion data
                x ~ uniform(-1, 1)
                t ~ uniform(0, 1)
            Parameters:
            -----------
                n: number of data point
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
        """
        x = torch.stack([
            torch_uniform(0, 1, [n]),
            torch_uniform(-1, 1, [n]),
        ], dim = 1)
        return x 
    
    def generate_test_data(self):
        """
            Load Test data from .mat file
                t in [0, 1], x in [-1, 1]
            Parameters:
            -----------
                n: number of data point
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([samples, n, y_dim])
        """
        if __name__ == '__main__':
            data = scipy.io.loadmat(".burgers_shock.mat")
        else:
            data = scipy.io.loadmat("equations/Burgers/burgers_shock.mat")
        x = torch.tensor(data["x"])      # [256, 1]
        t = torch.tensor(data["t"])      # [100, 1]
        y = torch.tensor(data["usol"].T) #[100, 245]
        
        self.ref_shape = y.shape

        x = torch.stack(torch.meshgrid(t[:,0], x[:,0]), -1).reshape(-1, self.x_dim)
        y = y[:, :, None].reshape(-1, self.y_dim)

        return x, y

    def pde_loss(self, y_pred):
        """
            the PDE operator
                ut + u ux - nu uxx = 0
            Parameters:
            -----------
                y_pred: torch.FloatTensor([n_f, y_dim])
            Returns:
            --------
                pde: torch.FloatTensor([n, y_dim])
        """
        dydx = gradient(y_pred, self.x_f_norm)
        dydxx = gradient(dydx, self.x_f_norm)
      
        ut = dydx[:, 0]
        ux = dydx[:, 1]
        uxx = dydxx[:, 1]
        # t, x = self.x_f_norm[:, 0], self.x_f_norm[:, 1]
        u    = y_pred[:, 0]
        jt, jx = self.jacobian[0, 0], self.jacobian[0, 1]
        # ut = grad(u, t)
        # ux = grad(u, x)
        # uxx = grad(ux, x)
        
        return mse(jt * ut + jx * u * ux - self.nu * jx**2 * uxx)

        
