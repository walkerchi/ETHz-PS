import torch 
import numpy as np
from ..base import Equation
from ..functional import gradient, partial_devirative, torch_uniform, torch_normal, mse, pi

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

    q = 8.25e-5
    u0 = -10
    Ks = 8.25e-4
    ug = 0
    m = 0.469
    alpha = 0.01
    L1 = 10
    L2 = 10
    ksat = 10

    @classmethod
    def s(cls,u):
        return (1 + (cls.alpha * (cls.ug - u))**(1/(1-cls.m)))**(-cls.m)

    @classmethod
    def K(cls,u):
        return cls.Ks * cls.s(u)**(1/2) * (1 - (1 - cls.s(u)**(1/cls.m))**cls.m)**2
    
    @classmethod
    def correct_y(cls, y):
        y[...,1] = cls.ksat * torch.exp(y[...,1])
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
        y += torch_normal(0, self.noise * y.std(0, keepdim=True),[n, 2])
        return x, y
    
    def generate_collosion_data(self, n = 1000):
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
        x1 = torch.stack([
            torch.full([n], self.L1),
            torch_uniform(0, self.L2, [n])
        ], -1)
        x2 = torch.stack([
            torch.zeros([n]),
            torch_uniform(0, self.L2, [n])
        ], -1)
        x3 = torch.stack([
            torch_uniform(0, self.L1, [n]),
            torch.zeros([n])
        ], -1)
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


    def pde_loss(self, y_pred, use_jacobian: bool = False):
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
        y_pred = self.correct_y(y_pred)

        dudx  = partial_devirative(y_pred, self.x_f_norm, y_index=0)
        dudxx = partial_devirative(dudx, self.x_f_norm)


        _, _, dudx_b2, dudx_b3, dudx_b4 = split(dudx, *[self.N_f - self.N_b * 4, *[self.N_b]*4])
        dudxx_f, _, _, _, _             = split(dudxx, *[self.N_f - self.N_b * 4, *[self.N_b]*4])

        y_f, y_b1, y_b2, _, _ = split(y_pred, *[self.N_f - self.N_b * 4, *[self.N_b]*4])
        # x_f, x_b1, x_b2, x_b3, x_b4 = split(self.x_f_norm, *[self.N_f - self.N_b * 4, *[self.N_b]*4])
       
        # u_f     = y_f[:, 0]
        k_f      =  y_f[:, 1]
        # x1_f    = x_f[:, 0]
        # x2_f    = x_f[:, 1]
        ux1x1_f = dudxx_f[:, 0]
        ux2x2_f = dudxx_f[:, 1]
        kux1x1_f = k_f * ux1x1_f
        kux2x2_f = k_f * ux2x2_f
        # ux1_f   = grad(u_f, x1_f)
        # ux2_f   = grad(u_f, x2_f)
        # kux1x1_f = grad(k_f*ux1_f, x1_f)
        # kux2x2_f = grad(k_f*ux2_f, x2_f)
        loss_f   = mse(kux1x1_f + kux2x2_f)

        u_b1    = y_b1[:, 0]
        loss_b1 = mse(u_b1 - self.u0)

        # x1_b2   = x_b2[:, 0]
        # u_b2    = y_b2[:, 0]
        k_b2      = y_b2[:, 1]
        # ux1_b2  = grad(u_b2, x1_b2)
        ux1_b2  = dudx_b2[:, 0]
        loss_b2 = mse(self.q + k_b2*ux1_b2)

        # x2_b3   = x_b3[:, 1]
        # u_b3    = y_b3[:, 0]
        # ux2_b3  = grad(u_b3, x2_b3)
        ux2_b3  = dudx_b3[:, 0]
        loss_b3 = mse(ux2_b3)

        # x2_b4   = x_b4[:, 1]
        # u_b4    = y_b4[:, 0]
        # ux2_b4  = grad(u_b4, x2_b4)
        ux2_b4  = dudx_b4[:, 0]
        loss_b4 = mse(ux2_b4)
        
        return loss_f + loss_b1 + loss_b2 + loss_b3 + loss_b4