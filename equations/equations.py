import numpy as np
import scipy.io
import torch
np.random.seed(1234)
torch.manual_seed(1234)
pi = np.pi


grad = lambda u, x: torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

class Equation:
    def __init__(self,
                N_f = 100,# number of collocation points
                N_u = 50, # number of training data 
                # N_ref = 200,# number of test data
                noise = 0.05,# noise level
                device = 'cpu'
                ):
        self.N_f = N_f
        self.N_u = N_u
        # self.N_ref = N_ref
        self.noise = noise
        self.device = device

        self.x_f                = self.generate_collosion_data(N_f)
        self.x_u, self.y_u      = self.generate_boundary_data(N_u)
        self.x_ref,self.y_ref   = self.generate_test_data()
        
        self.N_ref = len(self.x_ref)

        self.jacobian = 1 / (self.x_f.std(dim=0, keepdim=True))
        self.norm_x   = lambda x: (x - self.x_f.mean(dim=0, keepdim=True)) * self.jacobian
        
        self.x_f_norm = self.norm_x(self.x_f).requires_grad_(True)
        self.x_u_norm = self.norm_x(self.x_u).requires_grad_(True)

        self.to_device()

    def to_device(self):
        self.x_f = self.x_f.to(self.device)
        self.x_u = self.x_u.to(self.device)
        self.y_u = self.y_u.to(self.device)
        self.x_f_norm = self.x_f_norm.to(self.device)
        self.x_u_norm = self.x_u_norm.to(self.device)
        self.jacobian = self.jacobian.to(self.device)
        return self

    def cuda(self):
        self.device = 'cuda'
        self.to_device()
        return self
    
    def cpu(self):
        self.device = 'cpu'
        self.to_device()
        
    def generate_boundary_data(self, n):
        """
            Generate boundary data 

            Parameters:
            -----------
                n: number of data point
                
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([n, y_dim])
        """
        raise NotImplementedError()
    
    def generate_collosion_data(self, n):
        """
            Generate collosion data
    
            Parameters:
            -----------
                n: number of data point
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
        """
        raise NotImplementedError()
    
    def generate_test_data(self, n):
        """
            Generate test data
 
            Parameters:
            -----------
                n: number of data point
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([n, y_dim])
        """
        raise NotImplementedError()
    
    def f(self, x):
        """
            the applided external force
            Parameters:
            -----------
                x: torch.FloatTensor([n, x_dim])
            Returns:
            --------
                f: torch.FloatTensor([n, y_dim])
        """
        raise NotImplementedError()
    
    def pde(self, x, y, use_jacobian:bool=False):
        """
            the PDE operator
            Parameters:
            -----------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([n, y_dim])
            Returns:
            --------
                pde: torch.FloatTensor([n, y_dim])
        """
        raise NotImplementedError()
    
    def exact_solution(self, *args, **kwargs):
        """
            exact solution of `u` in the equation
            or `y` in the code
            Parameters:
            -----------
                x: torch.FloatTensor([n, x_dim])
            Returns:
            --------
                u: torch.FloatTensor([n, y_dim])
        """
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        return self.exact_solution(*args, **kwargs)

class ODE(Equation):
    """
    ODE
        uxx - u^2 ux = f(x)            x in [-1, 1]
        f(x) = -pi^2 sin(pi x) - pi cos(pi x) sin(pi x)^2
        u(-1) ~ normal(sin(pi), noise)
        u(1)  ~ normal(sin(pi), noise)
    """
    x_dim = 1 
    y_dim = 1
    has_exact_solution = True

    def generate_boundary_data(self, n):
        """
            Generate boundary data 
                u(-1) ~ normal(sin(pi), noise)
                u(1)  ~ normal(sin(pi), noise)

            Parameters:
            -----------
                n: number of data point
                
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([n, y_dim])
        """
        assert n % 2 == 0
        x = torch.tensor([-1.0, 1.0]).repeat(n//2)[:, None]
        y = self.exact_solution(x) + self.noise * torch.randn([n, 1])
        return x, y 
    
    def generate_collosion_data(self, n):
        """
            Generate collosion data
                x ~ uniform(-1, 1)
            Parameters:
            -----------
                n: number of data point
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
        """
        x = torch.linspace(-1, 1, n)[:, None]
        return x

    def generate_test_data(self):
        """
            Generate test data
                x ~ uniform(-1, 1)
            Parameters:
            -----------
                n: number of data point
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([n, y_dim])
        """
        if __name__ == "__main__":
            data = scipy.io.loadmat('ODE2000.mat')
        else:
            data = scipy.io.loadmat('equations/ODE2000.mat')
        y = torch.tensor(data["U"].reshape([-1, self.y_dim]))
        x = torch.arange(-1, 1.01, 0.01).tile(data["U"].shape[0], 1).reshape([-1, self.x_dim])
        return x, y

    def f(self, x):
        """
            the applided external force
                f(x) = -pi^2 sin(pi x) - pi cos(pi x) sin(pi x)^2
            Parameters:
            -----------
                x: torch.FloatTensor([n, x_dim])
            Returns:
            --------
                f: torch.FloatTensor([n, y_dim])
        """
        return -pi**2 * torch.sin(pi*x) - pi * torch.cos(pi * x) *  torch.sin(pi * x)**2

    def pde(self, u, x, use_jacobian=False):
        """
            the PDE operator
                uxx - u^2 ux
            Parameters:
            -----------
                u: torch.FloatTensor([n, y_dim])
                x: torch.FloatTensor([n, x_dim])
                use_jacobian: bool, whether to use jacobian
            Returns:
            --------
                pde: torch.FloatTensor([n, y_dim])
        """
        ux = grad(u, x)
        uxx = grad(ux, x)
        if use_jacobian:
            return self.jacobian**2 * uxx - self.jacobian * u**2 * ux
        else:
            return uxx - u**2 * ux
    
    def exact_solution(self, x):
        """
            exact solution of `u` in the equation
            or `y` in the code
                u = sin(pi * x)
            Parameters:
            -----------
                x: torch.FloatTensor([n, x_dim])
            Returns:
            --------
                u: torch.FloatTensor([n, y_dim])
        """
        return torch.sin(pi * x)



class Burgers(Equation):
    """
    Burgers
        x and t are all considered as dimension in x variable
        ut + u ux - nu uxx = 0      x in [-1, 1], t in [0, 1]
        u(0, x) = -sin(pi * x)
        u(t, -1) = u(t, 1) = 0
    """
    x_dim = 2  #[t, x]
    y_dim = 1
    has_exact_solution = False

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
        x = torch.uniform(-1, 1, [n // 3])
        e  = torch.randn([n // 3, 1]) / torch.exp(3 * torch.abs(x)) / self.noise
        x1 = torch.stack([t, x], dim=-1)
        y1 = - torch.sin(pi * (x + 2 * e))[:, None] + e
        
        x2 = torch.stack([
            torch.uniform(0, 1, [2 * n // 3]),
            torch.tensor([-1.0, 1.0]).repeat(n//3),
        ])
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
            torch.uniform(0, 1, [n]),
            torch.uniform(-1, 1, [n]),
        ], dim = 1)
        return x 
    
    def generate_test_data(self, n):
        """
            Generate test data
                x ~ uniform(-1, 1)
                t ~ uniform(0, 1)
            Parameters:
            -----------
                n: number of data point
            Returns:
            --------
                x: torch.FloatTensor([n, x_dim])
                y: torch.FloatTensor([n, y_dim])
        """
        data = scipy.io.loadmat(".burgers_shock.mat")
        x = torch.tensor(data["x"])
        t = torch.tensor(data["t"])
        y = torch.tensor(data["usol"])
        x = torch.stack(torch.meshgrid(x, t), -1).reshape([-1, self.x_dim])
        y = y.reshape([-1, self.y_dim])
        return x, y

    def pde(self, u, x, use_jacobian=False):
        """
            the PDE operator
                ut + u ux - nu uxx = 0
            Parameters:
            -----------
                u: torch.FloatTensor([n, y_dim])
                x: torch.FloatTensor([n, x_dim])
                use_jacobian: bool, whether to use jacobian
            Returns:
            --------
                pde: torch.FloatTensor([n, y_dim])
        """
        t, x = x 
        jt, jx = self.jacobian
        ut = grad(u, t)
        ux = grad(u, x)
        uxx = grad(ux, x)
        if use_jacobian:
            return jt * ut + jx * u * ux - jx**2 * uxx
        else:
            return ut + u * ux - uxx
        
    def f(self, x):
        """
            the applided external force
                f(x) = 0
            Parameters:
            -----------
                x: torch.FloatTensor([n, x_dim])
            Returns:
            --------
                f: torch.FloatTensor([n, y_dim])
        """
        return torch.zeros(x.shape[0], self.y_dim)
