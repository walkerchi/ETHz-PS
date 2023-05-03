import matplotlib.pyplot as plt 
import torch
import numpy as np
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set()


def plot_losses(losses, show:bool=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    for key, value in losses.items():
        ax.plot(value, label=key)
    ax.legend()
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.set_title("Losses", fontsize=20)
    
    if show:
        plt.show()
    return fig

def plot_x_y_uncertainty(equation, prediction, condition=None, show:bool=True):
    """
        Plot the uncertainty of the prediction on x_ref.
        Parameters:
        -----------
            equation: the equation
            prediction: the prediction of x_ref from the model
                torch.FloatTensor([samples, n_ref, y_dim]) or torch.FloatTensor([n_ref, y_dim])
                if the model is a Bayesian model, then `samples` is the number of samples
            condition: List[dict]
                condition on x_ref
            show: show the figure or not
        Returns:
        --------
            fig: the figure
    """
    if hasattr(equation, 'correct_y'):
        prediction = equation.correct_y(prediction)
    x_ref = equation.x_ref.detach().cpu().numpy() # [n, x_dim]
    y_ref = equation.y_ref.detach().cpu().numpy() # [samples, n, y_dim] or [n, y_dim]
    y_pre = prediction.detach().cpu().numpy()     # [samples, n, y_dim] or [n, y_dim]
    x_u   = equation.x_u.detach().cpu().numpy() # [n_u, x_dim]
    y_u   = equation.y_u.detach().cpu().numpy() # [n_u, y_dim]
    y_ref_has_samples = len(y_ref.shape) == 3
    y_pre_has_samples = len(y_pre.shape) == 3
    if y_ref_has_samples:
        y_ref_mu, y_ref_std = y_ref.mean(0), y_ref.std(0)
    if y_pre_has_samples:
        y_pre_mu, y_pre_std = y_pre.mean(0), y_pre.std(0)
    if condition is None:
        assert equation.x_dim == 1
        x_ref = x_ref[:, 0]
        fig ,ax = plt.subplots(ncols=equation.y_dim, figsize=(12, 6*equation.y_dim))
        def get_ax(i):
            if equation.y_dim == 1:
                return ax
            else:
                return ax[i]
        for i in range(equation.y_dim):

            if y_ref_has_samples:
                get_ax(i).plot(x_ref, y_ref_mu[:,i], 'b-', label='Exact')
                get_ax(i).fill_between(x_ref,
                                y_ref_mu[:, i]-2*y_ref_std[:, i],
                                y_ref_mu[:, i]+2*y_ref_std[:, i], 
                                facecolor='b', 
                                alpha=0.2, 
                                label='Two std band')
            else:
                get_ax(i).plot(x_ref, y_ref[:,i], 'b-', label='Exact')
            if y_pre_has_samples:
                get_ax(i).plot(x_ref, y_pre_mu[:,i], 'r--', label='Prediction')
                get_ax(i).fill_between(x_ref,
                                y_pre_mu[:, i]-2*y_pre_std[:, i],
                                y_pre_mu[:, i]+2*y_pre_std[:, i], 
                                facecolor='r', 
                                alpha=0.2, 
                                label='Two std band')
            else:
                get_ax(i).plot(x_ref, y_pre[:,i], 'r--', label='Prediction')
            get_ax(i).set_title(f"${equation.x_names[0]}$-${equation.y_names[i]}$")
            get_ax(i).set_xlabel(f"${equation.x_names[0]}$")
            get_ax(i).set_ylabel(f"${equation.y_names[i]}$")
            get_ax(i).scatter(x_u[:, 0], y_u[:, i], c='k', marker='x', label='Observations/BCs')
            get_ax(i).legend()

    else:
        fig, ax = plt.subplots(nrows=equation.y_dim, ncols=len(condition), figsize=(6*len(condition), 6*equation.y_dim),squeeze=False)
        for i in range(equation.y_dim):
            for j,cond in enumerate(condition):
                assert len(cond) + 1 == equation.x_dim
                mask = np.ones(x_ref.shape[0], dtype=bool)
                index = list(range(equation.x_dim))
                for k, v in cond.items():
                    x_ind = equation.x_names.index(k)
                    mask = mask & (x_ref[:, x_ind] == v)
                    index.remove(x_ind)
                index = index[0]

                if y_ref_has_samples:
                    ax[i, j].plot(x_ref[mask, index], y_ref_mu[mask,i], 'b-', label='Exact')
                    ax[i, j].fill_between(x_ref[mask],
                                    y_ref_mu[mask, i]-2*y_ref_std[mask, i],
                                    y_ref_mu[mask, i]+2*y_ref_std[mask, i], 
                                    facecolor='b', 
                                    alpha=0.2, 
                                    label='Two std band')
                else:
                    ax[i, j].plot(x_ref[mask, index], y_ref[mask,i], 'b-', label='Exact')

                if y_pre_has_samples:
                    ax[i, j].plot(x_ref[mask, index], y_pre_mu[mask, i], 'r--', label='Prediction')
                    ax[i, j].fill_between(x_ref[mask, index],
                                    y_pre_mu[mask, i]-2*y_pre_std[mask, i],
                                    y_pre_mu[mask, i]+2*y_pre_std[mask, i], 
                                    facecolor='r', 
                                    alpha=0.2, 
                                    label='Two std band')
                else:
                    ax[i, j].plot(x_ref[mask, index], y_pre[mask,i], 'r--', label='Prediction')
                ax[i, j].set_title(" ".join(f"${k}={v}$" for k,v in cond.items()))
                ax[i, j].set_xlabel(f"${equation.x_names[index]}$")
                ax[i, j].set_ylabel(f"${equation.y_names[i]}$")
                mask = np.ones(x_u.shape[0], dtype=bool)
                for k, v in cond.items():
                    x_ind = equation.x_names.index(k)
                    mask = mask & (x_u[:, x_ind] == v)
                ax[i, j].scatter(x_u[mask, index], y_u[mask, i], c='k', marker='x', label='Observations/BCs')
                ax[i, j].legend()        
    
    if show:
        plt.show()
    return fig

def plot_y_probability_given_x(equation, prediction, xs=None, show:bool=True):
    """
        Plot the probability of the prediction on xs.
        Parameters:
        -----------
            equation: the equation
            prediction: the prediction of x_ref from the model
                torch.FloatTensor([samples, n_ref, y_dim]) or torch.FloatTensor([n_ref, y_dim])
                if the model is a Bayesian model, then `samples` is the number of samples
            xs: the x values to plot the probability distribution of y_pred and y_ref
                list[torch.FloatTensor([x_dim])]
            show: show the figure or not
        Returns:
        --------
            fig: the figure
    """
    assert isinstance(xs, (list,tuple))
    if hasattr(equation, 'correct_y'):
        prediction = equation.correct_y(prediction)
    fig, ax = plt.subplots(nrows=equation.y_dim,ncols=len(xs), figsize=( len(xs) * 6 ,equation.y_dim*6))
    def get_ax(i,j):
        if equation.y_dim > 1 and len(xs) >1:
            return ax[i][j]
        elif len(xs) == 1 and equation.y_dim == 1:
            return ax
        elif len(xs) > 1:
            return ax[j]
        elif equation.y_dim > 1:
            return ax[i]
    y_pre = prediction.detach().cpu().numpy()     # [samples, n, y_dim] or [n, y_dim]
    x_ref = equation.x_ref.detach().cpu().numpy() # [n, x_dim]
    y_ref = equation.y_ref.detach().cpu().numpy() # [samples, n, y_dim] or [n, y_dim]
    for i in range(equation.y_dim):
        for j,x in enumerate(xs):
            if x in x_ref:
                where = np.where((x_ref==x).all(-1))[0] 
            else:
                where = np.array([np.abs(x_ref - x).argmin()])
            assert len(where) == 1
            sns.histplot(y_ref[:, where, i].flatten(), 
                        ax=get_ax(i,j), 
                        alpha=0.6, 
                        stat="probability", 
                        kde=True, 
                        color="b", 
                        element="step", 
                        label="Exact")
            sns.histplot(y_pre[:, where, i].flatten(),
                        ax=get_ax(i,j), 
                        color='r', 
                        alpha=0.6, 
                        stat="probability", 
                        kde=True,
                        element="step", 
                        label="Prediction")
            get_ax(i,j).set_xlabel(f"${equation.y_names[i]}({equation.x_names[i]}={x})$")
            get_ax(i,j).set_ylabel(f"$p(u)$")
            get_ax(i,j).set_title(f"${equation.y_names[i]}({equation.x_names[i]}={x})$ distribution")
            get_ax(i,j).legend()
    if show:
        plt.show()
    return fig
        
def plot_y_distribution_2D(equation, prediction, show:bool=True, align="row"):
    """
        Plot the 2D-distribution of the prediction and the reference.
        Parameters:
        -----------
            equation: the equation
            prediction: the prediction of x_ref from the model
                torch.FloatTensor([samples, n_ref, y_dim]) or torch.FloatTensor([n_ref, y_dim])
                if the model is a Bayesian model, then `samples` is the number of samples
            show: show the figure or not
        Returns:
        --------
            fig: the figure
    """
    assert align in ["row", "col"]
    assert equation.x_dim == 2
    assert len(equation.ref_shape) == 2
    if hasattr(equation, "correct_y"):
        prediction = equation.correct_y(prediction)
    x_u = equation.x_u.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    x_ref = equation.x_ref.detach().cpu().numpy()
    y_ref = equation.y_ref.detach().cpu().numpy()
    y_ref = y_ref.reshape([*equation.ref_shape, equation.y_dim])
    x1min, x1max, x2min, x2max = x_ref[:,0].min(), x_ref[:,0].max(), x_ref[:,1].min(), x_ref[:,1].max()
    if len(prediction.shape) == 3:
        prediction = prediction.reshape([-1, *equation.ref_shape, equation.y_dim])
        mu = prediction.mean(0)
        std = prediction.std(0)
        err = np.abs(y_ref - mu)
        if align == "row":
            fig, ax = plt.subplots(nrows=equation.y_dim, ncols=4, figsize=(20,5*equation.y_dim),squeeze=False)
        else:
            fig, ax = plt.subplots(ncols=equation.y_dim, nrows=4, figsize=(12*equation.y_dim,24),squeeze=False)
            ax      = ax.T
        for i in range(equation.y_dim):
            ax[i, 0].set_title(f"${equation.y_names[i]}$ exact")
            ax[i, 1].set_title(f"$\mu({equation.y_names[i]})$")
            ax[i, 2].set_title(f"$\sigma({equation.y_names[i]})$")
            ax[i, 3].set_title(f"$error({equation.y_names[i]})$")
            h = ax[i, 0].imshow(y_ref[:, :, i], 
                                interpolation='bicubic', 
                                cmap='rainbow',
                                extent=[x1min, x1max, x2min, x2max],
                                origin="lower",
                                aspect="auto")
            divider = make_axes_locatable(ax[i, 0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, orientation='vertical')
            h = ax[i, 1].imshow(mu[:, :, i], 
                                interpolation='bicubic', 
                                cmap='rainbow',
                                extent=[x1min, x1max, x2min, x2max],
                                origin="lower",
                                aspect="auto")
            divider = make_axes_locatable(ax[i, 1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, orientation='vertical')
            h = ax[i, 2].imshow(std[:, :, i], 
                                interpolation='bicubic', 
                                cmap='rainbow',
                                extent=[x1min, x1max, x2min, x2max],
                                origin="lower",
                                aspect="auto")
            divider = make_axes_locatable(ax[i, 2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, orientation='vertical')
            h = ax[i, 3].imshow(err[:, :, i], 
                                interpolation='bicubic', 
                                cmap='rainbow',
                                extent=[x1min, x1max, x2min, x2max],
                                origin="lower",
                                aspect="auto")
            divider = make_axes_locatable(ax[i, 3])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, orientation='vertical')
            for j in range(4):
                # ax[i, j].axis("off")
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].set_xlabel(f"${equation.x_names[0]}$")
                ax[i, j].set_ylabel(f"${equation.x_names[1]}$")
                ax[i, j].scatter(x_u[:,0], x_u[:,1], c='k', s=1, label=f"given data points")
                ax[i, j].legend()
                
    else:
        if align == "row":
            fig, ax = plt.subplots(nrows=equation.y_dim, ncols=3, figsize=(15,5*equation.y_dim),squeeze=False)
        else:
            fig, ax = plt.subplots(ncols=equation.y_dim, nrows=3, figsize=(12*equation.y_dim,18),squeeze=False)
            ax      = ax.T
        err = np.abs(y_ref - prediction)
        for i in range(equation.y_dim):
            ax[i, 0].set_title(f"${equation.y_names[i]}$ exact")
            ax[i, 1].set_title(f"${equation.y_names[i]}$ prediction")
            ax[i, 2].set_title(f"${equation.y_names[i]}$ error")
            h = ax[i, 0].imshow(y_ref[:, :, i], 
                                interpolation='bicubic', 
                                cmap='rainbow',
                                extent=[x1min, x1max, x2min, x2max],
                                origin="lower",
                                aspect="auto")
            divider = make_axes_locatable(ax[i, 0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, orientation='vertical')
            h = ax[i, 1].imshow(mu[:, :, i], 
                                interpolation='bicubic', 
                                cmap='rainbow',
                                extent=[x1min, x1max, x2min, x2max],
                                origin="lower",
                                aspect="auto")
            divider = make_axes_locatable(ax[i, 1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, orientation='vertical')
            h = ax[i, 2].imshow(err[:, :, i], 
                                interpolation='bicubic', 
                                cmap='rainbow',
                                extent=[x1min, x1max, x2min, x2max],
                                origin="lower",
                                aspect="auto")
            divider = make_axes_locatable(ax[i, 2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(h, cax=cax, orientation='vertical')
            for j in range(3):
                ax[i, j].axis("off")
                ax[i].set_xlabel(f"${equation.x_names[0]}$")
                ax[i].set_ylabel(f"${equation.x_names[1]}$")
                ax[i].scatter(x_u[:,0], x_u[:,1], c='k', s=1, label=f"given data points")
                ax[i].legend()
    if show:
        plt.show()
    return fig

def lineplot(x, y_exact, y_pred, x_points=None, y_points=None, title="", xlabel="", ylabel="", ax=None, show:bool=False):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().flatten()
    if isinstance(y_exact, torch.Tensor):
        y_exact = y_exact.detach().cpu().squeeze().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().squeeze().numpy()
    if ax is None:
        create_fig = True
        fig, ax = plt.subplots(figsize=(10,10))
    else:
        create_fig = False
    if len(y_exact.shape) == 2:
        mu, std  = y_exact.mean(axis=0), y_exact.std(axis=0)
        ax.plot(x, mu, label="exact", color="b", linestyle="-")
        ax.fill_between(x, mu-2 * std, mu+ 2 * std, alpha=0.3, color="b", label="two std band")
    else:
        assert len(y_exact.shape) == 1
        ax.plot(x, y_exact, label="exact", color="b", linestyle="-")

    if len(y_pred.shape) == 2:
        mu, std = y_pred.mean(axis=0), y_pred.std(axis=0)
        ax.plot(x, mu, label="prediction", color="r", linestyle="--")
        ax.fill_between(x, mu-2 * std, mu+ 2 * std, alpha=0.3, color="r", label="two std band")
    else:
        assert len(y_pred.shape) == 1
        ax.plot(x, y_pred, label="prediction", color="r", linestyle="--")
    
    if x_points is not None:
        assert y_points is not None
        ax.scatter(x_points, y_points, c="k", s=1, label="given data points")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

  
    if show:
        plt.show()
    if create_fig:
        return fig
    else:
        return ax    


