import argparse
import random
import os
import toml
import torch
import numpy as np

from models.uqpinn import UQPINN
import equations
from equations import ODE, Burgers
from plot import plot_uncertainty, plot_losses


def manul_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):

    manul_seed(args.seed)

    Equation = getattr(equations, args.equation)

    path = "output"
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, Equation.__name__)
    if not os.path.exists(path):
        os.mkdir(path)
    
    equation = Equation(N_u=args.n_boundary,
                        N_f=args.n_collosion,
                        noise=args.noise)
    
    pinn = UQPINN(x_dim=equation.x_dim, 
                  y_dim=equation.y_dim, 
                  z_dim=1,
                  n_layer_p=args.n_layer_p,
                  n_layer_q=args.n_layer_q,
                  n_layer_t=args.n_layer_t,
                  n_hidden_p=args.n_hidden_p,
                  n_hidden_q=args.n_hidden_q,
                  n_hidden_t=args.n_hidden_t,
                  lambd = args.lambd,
                  beta  = args.beta)
    
    device = torch.device("cuda" if args.device=='gpu' else "cpu")

    if args.eval:
        
        pinn.load_state_dict(torch.load(os.path.join(path, "ODE.pth"), map_location=device))
        losses = np.load(os.path.join(path, "losses.npz"), allow_pickle=True)
    
    else:
        
        if device == "cuda":
            equation.cuda()
            pinn.cuda()
        
        losses = pinn.fit(equation, epoch=args.epoch, k1=args.k1, k2=args.k2, print_every_epoch=args.log_every_epoch, lr=args.lr)

        torch.save(pinn.state_dict(), os.path.join(path, "ODE.pth"))
        np.savez(os.path.join(path, "losses.npz"), **losses)
        
    fig = plot_losses(losses, show=False)
    fig.savefig(os.path.join(path,"losses.png"))
    fig = plot_uncertainty(pinn, equation, n_samples=args.eval_n_samples, batch_size=args.eval_batch_size, show=False)
    fig.savefig(os.path.join(path,"uncertainty.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-eq', '--equation', type=str, default='ODE', choices=['ODE', 'Burgers'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('-on', '--device', type=str, default='gpu')
    parser.add_argument('--epoch', type=int, default=30000)
    parser.add_argument('--k1', type=int, default=1)
    parser.add_argument('--k2', type=int, default=5)
    parser.add_argument('--log_every_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-Nf', '--n_collosion', type=int, default=100)
    parser.add_argument('-Nu', '--n_boundary', type=int, default=100)
    parser.add_argument('-n', '--noise', type=float, default=0.05)
    parser.add_argument('--z_dim',  type=int, default=1)
    parser.add_argument('--n_layer_p', type=int, default=4)
    parser.add_argument('--n_layer_q', type=int, default=4)
    parser.add_argument('--n_layer_t', type=int, default=2)
    parser.add_argument('--n_hidden_p', type=int, default=50)
    parser.add_argument('--n_hidden_q', type=int, default=50)
    parser.add_argument('--n_hidden_t', type=int, default=50)
    parser.add_argument('--lambd', type=float, default=1.5)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--eval_n_samples', type=int, default=2000)
    args = parser.parse_args()

    if args.config is not None:
        with open(os.path.join("config", args.config + ".toml")) as f:
            config = toml.load(f)

        args.equation = config.get('equation', args.equation)
        args.eval = config.get('eval', args.eval)
        args.device = config.get('device', args.device)
        args.epoch = config.get('epoch', args.epoch)
        args.k1 = config.get('k1', args.k1)
        args.k2 = config.get('k2', args.k2)
        args.log_every_epoch = config.get('log_every_epoch', args.log_every_epoch)
        args.seed = config.get('seed', args.seed)
        args.n_collosion = config.get('n_collosion', args.n_collosion)
        args.n_boundary = config.get('n_boundary', args.n_boundary)
        args.noise = config.get('noise', args.noise)
        args.z_dim = config.get('z_dim', args.z_dim)
        args.n_layer_p = config.get('n_layer_p', args.n_layer_p)
        args.n_layer_q = config.get('n_layer_q', args.n_layer_q)
        args.n_layer_t = config.get('n_layer_t', args.n_layer_t)
        args.n_hidden_p = config.get('n_hidden_p', args.n_hidden_p)
        args.n_hidden_q = config.get('n_hidden_q', args.n_hidden_q)
        args.n_hidden_t = config.get('n_hidden_t', args.n_hidden_t)
        args.lambd = config.get('lambd', args.lambd)
        args.beta = config.get('beta', args.beta)
        args.lr = config.get('lr', args.lr)
    main(args)
