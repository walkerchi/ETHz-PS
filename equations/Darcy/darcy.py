import torch 
import numpy as np
from ..base import Equation
from ..functional import gradient, partial_derivative, torch_uniform, torch_normal, mse, pi

def split(tensor, *segments):
    assert sum(segments) == len(tensor)
    segments = list(segments)
    index = np.cumsum([0]+segments)
    return [tensor[index[i]:index[i+1]] for i in range(len(segments))]

class Darcy(Equation):
    """
        gradx (K(u) gradx u(x1,x2)) = 0         x in [0,L1]x[0,L2]
        u(x1, x2) = u0                          x1 = L1
        -K(u) du(x1, x2) / dx1 = q              x1 = 0  
        du(x1, x2)/dx2 = 0                      x2 = {0, L2}
        K(u) = K(s(u)) = Ks s^(1/2) (1 - (1 - s^(1/m))^m)^2
        s(u) = (1 + (alpha (ug - u))^(1/(1-m)))^(-m)

        q   = 8.25e-5 m/s 
        u0  = -10 m
        Ks  = 8.25e-4 m/s
        ug  = 0 
        m   = 0.469
        alpha = 0.01 
        L1  = 10m 
        L2  = 10m
    """
    x_dim = 2  # [x1, x2]
    y_dim = 2  # [u, K]
    x_names = ['x1', 'x2']
    y_names = ['u', 'K']
    x_boundary = [[0, 10], [0, 10]]
    has_exact_solution = False

    q = 1.
    u0 = -10
    Ks = 8.25e-4
    ug = 0
    m = 0.469
    alpha = 0.1
    L1 = 10
    L2 = 10

    ksat = 10

    @classmethod
    def s(cls,u):
        return (1 + (cls.alpha * torch.abs(cls.ug - u))**(1/(1-cls.m)))**(-cls.m)

    @classmethod
    def K(cls,u):
        s = cls.s(u)
        return torch.sqrt(s) * (1 - (1 - s**(1/cls.m))**cls.m)**2
    
    @classmethod
    def correct_y(cls, y):
        k_index = cls.y_names.index('K')
        y[...,k_index] = cls.ksat * torch.exp(y[...,k_index])
        return y
    
    @classmethod
    def correct_k(cls, k):
        return cls.ksat * torch.exp(k)

    def generate_boundary_data(self, n = 1000):
        """
            Generate boundary data 
                u(x1, x2) = u0                          x1 = L1
                -K(u) du(x1, x2) / dx1 = q              x1 = 0  
                du(x1, x2)/dx2 = 0                      x2 = {0, L2}
            Parameters:
            -----------
                n: number of data point
                
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([n, y_dim])
        """
        if __name__ == '__main__':
            data = np.load('nonlinear2d_data.npz')
        else:
            data = np.load('equations/Darcy/nonlinear2d_data.npz')
        x = data['X'] # [10000,2]
        k = data['k'] # [10000,1]
        u = data['u'] # [10000,1]
        index = np.random.choice(len(x), n, replace=False)
        x = x[index]
        u = u[index]
        k = k[index]
        x = torch.tensor(x)
        y = torch.tensor(np.concatenate([u,k], 1))
        y += torch_normal(0, self.noise * y.std(),[n, self.y_dim])
        return x, y
    
    def generate_collosion_data(self, n):
        """
            Generate collocation data
                x in [0,L1]x[0,L2]
                u(x1, x2) = u0                          x1 = L1
                -K(u) du(x1, x2) / dx1 = q              x1 = 0
                du(x1, x2)/dx2 = 0                      x2 = {0, L2}
            Parameters:
            -----------
                n: number of data point
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])

        """
        x = torch.stack([
            torch_uniform(0, self.L1, [n]),
            torch_uniform(0, self.L2, [n])
        ], -1)


        self.N_b = 100 
        n = self.N_b
        # u(x1, x2) = u0                x1 = L1, x2 in [0,L2]
        x1 = torch.stack([
            torch.full([n], self.L1),
            torch_uniform(0, self.L2, [n])
        ], -1)
        # # -K(u) du(x1, x2) / dx1 = q    x1 = 0, x2 in [0,L2]
        x2 = torch.stack([
            torch.zeros([n]),
            torch_uniform(0, self.L2, [n])
        ], -1)
        # du(x1, x2)/dx2 = 0            x2 = 0, x1 in [0,L1]
        x3 = torch.stack([
            torch_uniform(0, self.L1, [n]),
            torch.zeros([n])
        ], -1)
        # du(x1, x2)/dx2 = 0            x2 = L2, x1 in [0,L1]
        x4 = torch.stack([
            torch_uniform(0, self.L1, [n]),
            torch.full([n], self.L2)
        ], -1)

        x = torch.cat([x, x1, x2, x3, x4], dim=0)

        return x    
    
    def generate_test_data(self):
        if __name__ == '__main__':
            data = np.load('nonlinear2d_data.npz')
        else:
            data = np.load('equations/Darcy/nonlinear2d_data.npz')
        x = data['X'] # [10000,2]
        k = data['k'] # [10000,1]
        u = data['u'] # [10000,1]

        x = torch.tensor(x)
        y = torch.tensor(np.concatenate([u,k], 1))

        self.ref_shape = [100, 100]

        return x, y


    def pde_loss(self, y_pred):
        """
            gradx (K(u) gradx u(x1,x2)) = 0         x in [0,L1]x[0,L2]
            u(x1, x2) = u0                          x1 = L1
            -K(u) du(x1, x2) / dx1 = q              x1 = 0
            du(x1, x2)/dx2 = 0                      x2 = {0, L2}
            Parameters:
            -----------
                y_pred: torch.FloatTensor([n, y_dim])
                use_jacobian: bool
            Returns:
            --------
                torch.FloatTensor([1])        
        """
        x1_index = self.x_names.index('x1')
        x2_index = self.x_names.index('x2')
        u_index  = self.y_names.index('u')
        k_index  = self.y_names.index('K')

        y_pred = self.correct_y(y_pred)
        N_f    = self.N_f - self.N_b * 4
        ux  = partial_derivative(y_pred, self.x_f_norm, y_index=u_index)

        ux_f, _, ux_b2, ux_b3, ux_b4 = split(ux, *[N_f, *[self.N_b]*4])
    
        y_f, y_b1, y_b2, _, _ = split(y_pred, *[N_f,  *[self.N_b]*4])
        # x_f, x_b1, x_b2, x_b3, x_b4 = split(self.x_f_norm, *[self.N_f - self.N_b * 4, *[self.N_b]*4])
       
        # gradx (K(u) gradx u(x1,x2)) = 0
        k_f     =  y_f[:, k_index]
        ux1_f   = ux_f[:, x1_index]
        ux2_f   = ux_f[:, x2_index]
        kux1_f  = k_f * ux1_f
        kux2_f  = k_f * ux2_f
        kux1x1_f = partial_derivative(kux1_f, self.x_f_norm, x_index=x1_index)[:N_f]
        kux2x2_f = partial_derivative(kux2_f, self.x_f_norm, x_index=x2_index)[:N_f]
        loss_f   = mse(kux1x1_f + kux2x2_f)

        # u(x1, x2) = u0                x1 = L1
        u_b1    = y_b1[:, u_index]
        loss_b1 = mse(u_b1 - self.u0)

        # -K(u) du(x1, x2) / dx1 = q    x1 = 0
        k_b2      = y_b2[:, k_index]
        ux1_b2  = ux_b2[:, x1_index]
        loss_b2 = mse(self.q + k_b2*ux1_b2)

        # du(x1, x2)/dx2 = 0            x2 = 0    
        ux2_b3  = ux_b3[:, x2_index]
        loss_b3 = mse(ux2_b3)

        # du(x1, x2)/dx2 = 0            x2 = L2
        ux2_b4  = ux_b4[:, x2_index]
        loss_b4 = mse(ux2_b4)
        
        return loss_f + loss_b1 + loss_b2 + loss_b3 + loss_b4