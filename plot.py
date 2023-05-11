import matplotlib.pyplot as plt 
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import tikzplotlib as tikz
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use("ggplot")


USE_TEX = False
class Figure:
    def __init__(self, fig):
        self.fig = fig
    def savefig(self, filename):
        if USE_TEX:
            for image_name in ["png", "jpg", "jpeg", "pdf", "svg"]:
                if filename.endswith(image_name):
                    filename = filename.replace(image_name, "tex")
            tikz.save(filename)
        else:
            self.fig.savefig(filename)

sns.set()

def _lineplot(x, y, ax, label="", color=None, linestyle="-"):
    """
        Parameters:
        -----------
            x: torch.Tensor [n]
            y: torch.Tensor [n] or [samples, n]
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().flatten()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().squeeze().numpy()
    if len(y.shape) == 2:
        mu, std = y.mean(axis=0), y.std(axis=0)
        ax.plot(x, mu, label=label, color=color, linestyle=linestyle)
        ax.fill_between(x, mu-2 * std, mu+ 2 * std, alpha=0.3, color=color, label=f"$2\sigma({label})$")
    else:
        assert len(y.shape) == 1
        ax.plot(x, y, label=label, color=color, linestyle=linestyle)

def _heatmap(data, fig, ax, extent, xabel="", ylabel="", title=""):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    h = ax.imshow(data.T, 
                        interpolation='bicubic', 
                        cmap='rainbow',
                        extent=extent,
                        origin="lower",
                        aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def _errormaps(exact, pred, fig, ax, extent, xlabel="", ylabel="",prefix="", title=""):
    if isinstance(exact, torch.Tensor):
        exact = exact.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if len(pred.shape) == 3:
        assert len(ax) >= 4 
        mu, std = pred.mean(0), pred.std(0)
        error   = (exact - mu)
        _heatmap(exact, fig, ax[0], extent, xlabel, ylabel, prefix+f"${title}$ Exact")
        _heatmap(mu, fig, ax[1], extent, xlabel, ylabel, prefix+f"$\mu({title})$")
        _heatmap(std, fig, ax[2], extent, xlabel, ylabel, prefix+f"$\sigma({title})$")
        _heatmap(error, fig, ax[3], extent, xlabel, ylabel, prefix+"error")
    elif len(pred.shape) == 2:
        assert len(ax) >= 3
        error = (exact - pred)
        _heatmap(exact, fig, ax[0], extent, xlabel, ylabel, prefix+f"${title}$ Exact")
        _heatmap(pred, fig, ax[1], extent, xlabel, ylabel, prefix+f"${title}$ Prediction")
        _heatmap(error, fig, ax[2], extent, xlabel, ylabel, prefix+"error")
    else:
        raise NotImplementedError()
    

def plot_losses(losses, show:bool=True):
    """
        losses: Dict[str, List[float]]
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for key, value in losses.items():
        ax.plot(value, label=key)
    ax.legend()
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.set_title("Losses", fontsize=20)
    
    if show:
        plt.show()
    return Figure(fig)

def plot_x_y_uncertainty(equation, prediction, condition=None, show:bool=True):
    """
        Plot the uncertainty of the prediction on x_ref.
        Parameters:
        -----------
            equation: the equation
            prediction: the prediction of x_ref from the model
                torch.FloatTensor([samples, n_ref, y_dim]) or torch.FloatTensor([n_ref, y_dim])
                if the model is a Bayesian model, then `samples` is the number of samples

                of Dict[str, torch.FloatTensor] 
            condition: List[dict]
                condition on x_ref
            show: show the figure or not
        Returns:
        --------
            fig: the figure
    """
    if hasattr(equation, 'correct_y'):
        if isinstance(prediction, dict):
            for key, value in prediction.items():
                prediction[key] = equation.correct_y(value)
        else:
            prediction = equation.correct_y(prediction)
    x_ref = equation.x_ref
    y_ref = equation.y_ref
    y_pre = prediction
    x_u   = equation.x_u.detach().cpu().numpy() # [n_u, x_dim]
    y_u   = equation.y_u.detach().cpu().numpy() # [n_u, y_dim]

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
            if isinstance(y_pre, dict):
                _lineplot(x_ref, y_ref[...,i], get_ax(i), label="Exact", color=None, linestyle="-")
                for key, value in y_pre.items():
                    _lineplot(x_ref, value[...,i], get_ax(i), label=key, color=None, linestyle="--")
            else:
                _lineplot(x_ref, y_ref[...,i], get_ax(i), label="Exact", color="b", linestyle="-")
                _lineplot(x_ref, y_pre[...,i], get_ax(i), label="Prediction", color="r", linestyle="--")
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
                if isinstance(y_pre, dict):
                    _lineplot(x_ref[mask, index], y_ref[...,mask, i], ax[i, j], label="Exact", color=None, linestyle="-")
                    for key, value in y_pre.items():
                        _lineplot(x_ref, value[...,mask,i], ax[i, j], label=key, color=None, linestyle="--")
                else:
                    _lineplot(x_ref, y_ref[...,mask,i], ax[i, j], label="Exact", color="b", linestyle="-")
                    _lineplot(x_ref, y_pre[...,mask,i], ax[i, j], label="Prediction", color="r", linestyle="--")

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
    return Figure(fig)

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
    return Figure(fig)

def plot_y_distribution_2D(equation, prediction, show:bool=True, align="row", return_ax:bool=False):
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
    if hasattr(equation, 'correct_y'):
        if isinstance(prediction, dict):
            for key, value in prediction.items():
                prediction[key] = equation.correct_y(value)
        else:
            prediction = equation.correct_y(prediction)
    x_u = equation.x_u.detach().cpu().numpy()
    x_ref = equation.x_ref.detach().cpu().numpy()
    y_ref = equation.y_ref
    y_ref = y_ref.reshape([*equation.ref_shape, equation.y_dim])
    extent = x_ref[:,0].min(), x_ref[:,0].max(), x_ref[:,1].min(), x_ref[:,1].max()
    

    if isinstance(prediction, dict):
        y_pre = {}
        line_size = 3
        n = len(prediction) if isinstance(prediction, dict) else 1
        for key, value in prediction.items():
            if len(value.shape) == 3:
                y_pre[key] = value.reshape([-1, *equation.ref_shape, equation.y_dim])
                line_size = 4
            elif len(value.shape) == 2:
                y_pre[key] = value.reshape([*equation.ref_shape, equation.y_dim])
            else:
                raise NotImplementedError()
        if align == "row":
            fig, ax = plt.subplots(facecolor="white", nrows=equation.y_dim*n, ncols=line_size, figsize=(5*line_size,5*equation.y_dim),squeeze=False)
        else:
            fig, ax = plt.subplots(facecolor="white",ncols=equation.y_dim*n, nrows=line_size, figsize=(12*equation.y_dim,6*line_size),squeeze=False)
            ax      = ax.T
        for i in range(0, equation.y_dim*n, n):
            for j in range(n):
                key = list(prediction.keys())[j]
                _errormaps(y_ref[...,i], 
                    y_pre[key][..., i], 
                    fig,
                    ax=ax[i+j], 
                    extent=extent,
                    prefix =key+" ",
                    title=equation.y_names[i], 
                    xlabel=f"${equation.x_names[0]}$", 
                    ylabel=f"${equation.x_names[1]}$")
        
                for k in range(len(y_pre[key].shape)):
                    ax[i+j, k].scatter(x_u[:,0], x_u[:,1], c='k', s=1, label=f"given data points")
                    ax[i+j, k].legend()
                for k in range(len(y_pre[key].shape), line_size):
                    ax[i+j, k].axis("off")
                    
    else:
        if len(prediction.shape) == 3:
            line_size = 4 
            prediction = prediction.reshape([-1, *equation.ref_shape, equation.y_dim])
        elif len(prediction.shape) == 2:
            line_size = 3
            prediction = prediction.reshape([*equation.ref_shape, equation.y_dim])
        else:
            raise NotImplementedError()
        if align == "row":
            fig, ax = plt.subplots(nrows=equation.y_dim, ncols=4, figsize=(20,5*equation.y_dim),squeeze=False)
        else:
            fig, ax = plt.subplots(ncols=equation.y_dim, nrows=4, figsize=(12*equation.y_dim,24),squeeze=False)
            ax      = ax.T
        for i in range(0,equation.y_dim):
            _errormaps(y_ref[...,i], 
                prediction[..., i], 
                fig,
                ax=ax[i], 
                extent=extent, 
                title=equation.y_names[i], 
                xlabel=f"${equation.x_names[0]}$", 
                ylabel=f"${equation.x_names[1]}$")
            for j in range(len(prediction.shape)):
                ax[i, j].scatter(x_u[:,0], x_u[:,1], c='k', s=1, label=f"given data points")
                ax[i, j].legend()
            
    if show:
        plt.show()
    if return_ax:
        return Figure(fig), ax
    return Figure(fig)

def lineplot(x, y_exact, y_pred, x_points=None, y_points=None, title="", xlabel="", ylabel="", show:bool=False):
    """
        Parameters:
        ----------
        x: torch.Tensor or numpy.ndarray [n]
            x-axis values
        y_exact: torch.Tensor or numpy.ndarray [n] or [samples, n]
            exact y-axis values
        y_pred: torch.Tensor or numpy.ndarray [n] or [samples, n] or Dict[str, torch.Tensor or numpy.ndarray [n] or [samples, n]]
            predicted y-axis values
        x_points: torch.Tensor or numpy.ndarray [m]
            x-axis values of given data points
        y_points: torch.Tensor or numpy.ndarray [m] or [samples, m]
            y-axis values of given data points
        title: str
            title of the plot
        xlabel: str
            label of the x-axis
        ylabel: str
            label of the y-axis
        show: bool
            whether to show the plot

        Returns:
        ----------
        fig: matplotlib.figure.Figure

    """
    # if isinstance(x, torch.Tensor):
    #     x = x.detach().cpu().numpy().flatten()
    # if isinstance(y_exact, torch.Tensor):
    #     y_exact = y_exact.detach().cpu().squeeze().numpy()
    # if isinstance(y_pred, torch.Tensor):
    #     y_pred = y_pred.detach().cpu().squeeze().numpy()
    # if len(y_exact.shape) == 2:
    #     mu, std  = y_exact.mean(axis=0), y_exact.std(axis=0)
    #     ax.plot(x, mu, label="exact", color="b", linestyle="-")
    #     ax.fill_between(x, mu-2 * std, mu+ 2 * std, alpha=0.3, color="b", label="two std band")
    # else:
    #     assert len(y_exact.shape) == 1
    #     ax.plot(x, y_exact, label="exact", color="b", linestyle="-")

    # if len(y_pred.shape) == 2:
    #     mu, std = y_pred.mean(axis=0), y_pred.std(axis=0)
    #     ax.plot(x, mu, label="prediction", color="r", linestyle="--")
    #     ax.fill_between(x, mu-2 * std, mu+ 2 * std, alpha=0.3, color="r", label="two std band")
    # else:
    #     assert len(y_pred.shape) == 1
    #     ax.plot(x, y_pred, label="prediction", color="r", linestyle="--")
    
    fig, ax = plt.subplots(figsize=(10,10))

    if isinstance(y_pred, dict):
        _lineplot(x, y_exact, ax= ax, label="exact", color=None, linestyle="--")
        for k,v in y_pred.items():
            _lineplot(x, v, ax= ax, label=f"{k} prediction", color=None, linestyle="--")
    else:
        _lineplot(x, y_exact, ax= ax, label="exact", color="b", linestyle="--")
        _lineplot(x, y_pred, ax= ax, label="prediction", color="r", linestyle="--")

    if x_points is not None:
        assert y_points is not None
        if isinstance(x_points, torch.Tensor):
            x_points = x_points.detach().cpu().numpy().flatten()
        if isinstance(y_points, torch.Tensor):
            y_points = y_points.detach().cpu().numpy().flatten()
        ax.scatter(x_points, y_points, c="k", s=1, label="given data points")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

  
    if show:
        plt.show()
    return Figure(fig) 
    



