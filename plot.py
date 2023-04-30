import matplotlib.pyplot as plt 
import numpy as np



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

def plot_uncertainty(model,equation, n_samples:int, batch_size:int, show:bool=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    mu,std = model.predict(equation, equation.x_ref, n_samples, batch_size)
    mu = mu.detach().cpu().numpy()
    std = std.detach().cpu().numpy()
    x_u = equation.x_u.detach().cpu().numpy()
    y_u = equation.y_u.detach().cpu().numpy()
    x_ref = equation.x_ref.detach().cpu().numpy()
    y_ref = equation.y_ref.detach().cpu().numpy()
    ax.plot(x_u, y_u, 'kx', label='Boundary Condition', markersize=4)
    ax.scatter(x_ref, y_ref, label='Exact')
    ax.scatter(x_ref, mu, label='Prediction')
    ax.fill_between(x_ref.flatten(), mu.flatten()-2*std.flatten(), mu.flatten()+2*std.flatten(), facecolor='orange', alpha=0.5, label='Two std band')
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("$y$", fontsize=20)
    error = np.linalg.norm(y_ref-mu) / np.linalg.norm(y_ref)
    ax.set_title(f"Error:{error}")
    ax.legend(loc='upper left')
    
    if show:
        plt.show()
    return fig


