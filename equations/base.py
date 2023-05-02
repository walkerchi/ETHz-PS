import torch
import numpy as np
pi   = np.pi
grad = lambda u, x: torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]



class Equation:
    x_dim = 1 
    y_dim = 1
    x_names = ['x']
    y_names = ['u']
    has_exact_solution = False
    def __init__(self,
                N_f = 100,# number of collocation points
                N_u = 50, # number of training data 
                noise = 0.05,# noise level
                device = 'cpu'
                ):
        assert len(self.x_names) == self.x_dim
        assert len(self.y_names) == self.y_dim

        # self.N_f = N_f
        self.N_u = N_u
        self.noise = noise
        self.device = device

        self.x_f                = self.generate_collosion_data(N_f)
        self.x_u, self.y_u      = self.generate_boundary_data(N_u)
        self.x_ref,self.y_ref   = self.generate_test_data()
       
        self.N_ref = len(self.x_ref)
        self.N_f   = len(self.x_f)
        
        x_f_mean = self.x_f.mean(dim=0, keepdim=True)
        x_f_std  = self.x_f.std(dim=0, keepdim=True)
        self.jacobian = 1 / x_f_std
        self.norm_x   = lambda x: (x - self.x_f.mean(dim=0, keepdim=True)) * self.jacobian
        self.unnorm_x = lambda x: x * x_f_std + x_f_mean

        self.x_f_norm = self.norm_x(self.x_f).requires_grad_(True)
        self.x_u_norm = self.norm_x(self.x_u).requires_grad_(True)

        self.to_device()

    def to_device(self):
        self.x_f = self.x_f.to(self.device)
        self.x_u = self.x_u.to(self.device)
        self.y_u = self.y_u.to(self.device)
        self.x_ref = self.x_ref.to(self.device)
        self.y_ref = self.y_ref.to(self.device)
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
                y: torch.FloatTensor([samples, n, y_dim]) or torch.FloatTensor([n, y_dim])
        """
        raise NotImplementedError()
    
    # def f(self, x):
    #     """
    #         the applided external force
    #         Parameters:
    #         -----------
    #             x: torch.FloatTensor([n, x_dim])
    #         Returns:
    #         --------
    #             f: torch.FloatTensor([n, y_dim])
    #     """
    #     raise NotImplementedError()
    
    # def pde(self, x, y, use_jacobian:bool=False):
    #     """
    #         the PDE operator
    #         Parameters:
    #         -----------
    #             x: torch.FloatTensor([n, x_dim])
    #             y: torch.FloatTensor([n, y_dim])
    #         Returns:
    #         --------
    #             pde: torch.FloatTensor([n, y_dim])
    #     """
    #     raise NotImplementedError()
    
    def pde_loss(self, y_f_pred):
        """
            the PDE operator
            Parameters:
            -----------
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
