import torch 
import torch.nn as nn 
import numpy as np
from tqdm import tqdm 
from .utils import StackMLP, MLP


class PINN(nn.Module):
    def __init__(self, nn, n_hidden, n_layer, x_dim, y_dim, k_dim,**kwargs):
        super().__init__()
        self.k_dim = k_dim
        self.actual_y_dim = y_dim if k_dim is None else y_dim - k_dim
        self.nn = nn(x_dim, y_dim, n_hidden, n_layer)
    @property
    def device(self):
        return next(self.parameters()).device
    def loss(self, equation):
        x_u, y_u = equation.x_u_norm, equation.y_u
        x_f      = equation.x_f_norm
        y_u_pred = self.nn(x_u)[:,:self.actual_y_dim]
        y_f_pred = self.nn(x_f)
        regression_loss = torch.mean((y_u_pred - y_u)**2)
        pde_loss = equation.pde_loss(y_f_pred)
        return regression_loss + pde_loss, regression_loss, pde_loss

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
    
    def anim_fit(self, equation, epoch=2000, print_every_epoch=100, lr=1e-4, **kwargs):
        import matplotlib.pyplot as plt 
        from matplotlib.animation import FuncAnimation
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = {"loss":[]}

        fig, ax = plt.subplots(figsize=(12,6),ncols=2)
        pde_loss = []
        reg_loss = []

        def draw():
            with torch.no_grad():
                ax[0].clear()
                ax[1].clear()
                x = equation.x_f_norm
                y = self.nn(x)
                u = y[:,0].cpu().numpy()
                k = y[:,1].cpu().numpy()
                index = np.argsort(u)
                u = u[index]
                k = np.exp(k[index])
                line = ax[0].plot(u, k)
                ax[0].set_xlabel("u")
                ax[0].set_ylabel('k')
                ax[0].set_title("u-k")
                ax[0].set_xlim(left=-10,right=10)
                ax[0].set_ylim(bottom=-1, top=1)
                ax[1].set_xlabel("epoch")
                ax[1].set_ylabel("loss")
                ax[1].plot(pde_loss, label="pde_loss", color="lightblue")
                ax[1].plot(reg_loss, label="regression_loss", color="orange")
                ax[1].set_ylim(top=50)
                if len(pde_loss) > 0:
                    ax[1].text(len(pde_loss),pde_loss[-1],str(pde_loss[-1]), color="lightblue")
                    ax[1].text(len(reg_loss),reg_loss[-1],str(reg_loss[-1]), color="orange")
                ax[1].legend()
            return line

        draw()
        
        def update(ep):
            optimizer.zero_grad()
            loss, reg_l, pde_l = self.loss(equation)
            loss.backward()
            optimizer.step()
            pde_loss.append(pde_l.item())
            reg_loss.append(reg_l.item())
            if (ep+1)%print_every_epoch == 0:
                self.eval()
                loss,_,_ = self.loss(equation)
                losses["loss"].append(loss.item())
                self.train()
            return draw()
       
        anim = FuncAnimation(fig, 
                            update,
                            init_func=draw, 
                             frames=tqdm(range(epoch), unit="epoch"))
        anim.save("u-k.mp4", fps=10, writer="ffmpeg")
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
        if self.k_dim is not None:
            y = equation.correct_y(y)
        return y