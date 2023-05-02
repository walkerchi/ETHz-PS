
import torch 
import scipy.io
from ..base import Equation
from ..functional import gradient ,pi, mse


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
    x_names = ['x']
    y_names = ['u']
    x_boundary = [[-1,1]]
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
                y: torch.FloatTensor([samples, n, y_dim])
        """
        if __name__ == "__main__":
            data = scipy.io.loadmat('ODE2000.mat')
        else:
            data = scipy.io.loadmat('equations/ODE/ODE2000.mat')
        y = torch.tensor(data["U"])[:, :,  None]
        x = torch.arange(-1, 1.01, 0.01)[:, None]
        return x, y
        # x = torch.linspace(-1, 1, 100)[:, None]
        # y = self.exact_solution(x)
        # return x, y

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
        x = self.unnorm_x(x)
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
        ux = gradient(u, x)
        uxx = gradient(ux, x)
        if use_jacobian:
            return self.jacobian**2 * uxx - self.jacobian * u**2 * ux
        else:
            return uxx - u**2 * ux
    
    def pde_loss(self, u, x):
        return mse(self.pde(u,x, True) - self.f(x))

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

