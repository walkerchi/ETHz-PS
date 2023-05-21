import argparse
import random
import os
import toml
import torch
import numpy as np

from models import UQPINN, MLP, StackMLP, PINN
import equations
from equations import ODE, Burgers
from plot import plot_losses, plot_x_y_uncertainty, plot_y_probability_given_x, plot_y_distribution_2D, lineplot, plot_u_k_relation


def manul_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def build_model(args):
    if isinstance(args.nn, str):
        args.nn = {
            "MLP":MLP,
            "StackMLP":StackMLP
        }[args.nn]
    if args.equation == "Darcy":
        args.nn = StackMLP
    Equation = getattr(equations, args.equation)
    model =  {
        "uqpinn":UQPINN,
        "pinn":PINN
    }[args.model](x_dim=Equation.x_dim, 
        y_dim=Equation.y_dim, 
        z_dim=1,
        n_layer=args.n_layer,
        n_layer_q=args.n_layer_q,
        n_layer_t=args.n_layer_t,
        n_hidden=args.n_hidden,
        n_hidden_q=args.n_hidden_q,
        n_hidden_t=args.n_hidden_t,
        nn  = args.nn,
        k_dim = Equation.k_dim if hasattr(Equation, "k_dim") else None,
        lambd = args.lambd,
        beta  = args.beta)
    return model

def build_equation(args):
    Equation = getattr(equations, args.equation)
    equation = Equation(N_u=args.n_boundary,
                        N_f=args.n_collosion,
                        noise=args.noise)
    return equation

def main(args):

    manul_seed(args.seed)

    # Equation = getattr(equations, args.equation)

    path = "output"
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, args.config)
    if not os.path.exists(path):
        os.mkdir(path)
    
    equation = build_equation(args)
    pinn = build_model(args)
    # args.nn = {
    #     "MLP":MLP,
    #     "StackMLP":StackMLP
    # }[args.nn]
    # if args.equation == "Darcy":
    #     args.nn = StackMLP

    # equation = Equation(N_u=args.n_boundary,
    #                     N_f=args.n_collosion,
    #                     noise=args.noise)
    
    # pinn = {
    #     "uqpinn":UQPINN,
    #     "pinn":PINN
    # }[args.model](x_dim=equation.x_dim, 
    #     y_dim=equation.y_dim, 
    #     z_dim=1,
    #     n_layer=args.n_layer,
    #     n_layer_q=args.n_layer_q,
    #     n_layer_t=args.n_layer_t,
    #     n_hidden=args.n_hidden,
    #     n_hidden_q=args.n_hidden_q,
    #     n_hidden_t=args.n_hidden_t,
    #     nn  = args.nn,
    #     lambd = args.lambd,
    #     beta  = args.beta)

    
    if args.eval:
        
        pinn.load_state_dict(torch.load(os.path.join(path,f"weight.pth"), map_location="cuda:0" if args.device=="gpu" else "cpu"))
        losses = np.load(os.path.join(path, "losses.npz"), allow_pickle=True)
    
    else:
        
        if args.device == "gpu":
            equation.cuda()
            pinn.cuda()
    
        losses = pinn.fit(equation, epoch=args.epoch, k1=args.k1, k2=args.k2, print_every_epoch=args.log_every_epoch, lr=args.lr)

        torch.save(pinn.state_dict(), os.path.join(path, f"weight.pth"))
        np.savez(os.path.join(path, "losses.npz"), **losses)
 
    prediction = pinn.predict(equation, n_samples=args.eval_n_samples, batch_size=args.eval_batch_size)
    fig = plot_losses(losses, show=False)
    fig.savefig(os.path.join(path,"losses.png"))
    if args.equation == "ODE":
        lineplot(
            equation.x_ref, 
            equation.y_ref,
            prediction,
            xlabel="$x$", 
            ylabel="$u(x)$",
            title =f"{args.model}-{args.equation}" ,
            x_points=equation.x_u,
            y_points=equation.y_u,
            show=False).savefig(os.path.join(path,"x_y_relation_2D.png"))
        # plot_x_y_uncertainty(equation, prediction, show=False).savefig(os.path.join(path,"x_y_uncertainty.png"))
        if prediction.dim() == 3:
            plot_y_probability_given_x(equation, prediction,(-0.5, 0.5), show=False).savefig(os.path.join(path,"y_probability_given_x.png"))
    elif args.equation == "Burgers":
        plot_y_distribution_2D(equation, prediction, align="col",show=False).savefig(os.path.join(path,"y_distribution_2D.png"))
        plot_x_y_uncertainty(equation, prediction, [{"t":0.25},{"t":0.5},{"t":0.75}], show=False).savefig(os.path.join(path,"x_y_uncertainty.png"))
    elif args.equation == "Darcy":
        plot_y_distribution_2D(equation, prediction, align="row", show=False).savefig(os.path.join(path,"y_distribution_2D.png"))
        with torch.no_grad():
            u = torch.linspace(-10, -4, 100)[:, None].to(pinn.device)
            k_exact= equation.K(u)
        plot_u_k_relation(prediction, u, k_exact, show=False).savefig(os.path.join(path,"y_relation_2D.png"))
        # lineplot(u, k_exact, k_pred, 
        #          x_points= equation.y_u[:,0],
        #          y_points= equation.k_u[:,0],
        #          xlabel="$u$", ylabel="$K(u)$", show=False).savefig(os.path.join(path,"y_relation_2D.png"))

def compare(args):
    manul_seed(args.seed)
    targets = ["pinn", "uqpinn"]
    predictions = {}
    losses = {}
    models = []
    equation = build_equation(args)
    for target in targets:
        args.model = target
        model = build_model(args)
        models.append(model)
        path = os.path.join("output", f"{args.equation}_{target}", "weight.pth")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        loss = np.load(os.path.join("output", f"{args.equation}_{target}", "losses.npz"), allow_pickle=True)
        loss = dict(loss)
        for k,v in loss.copy().items():
            loss[f"{target} {k}"] = loss.pop(k)
        losses = {**losses, **loss}
        prediction = model.predict(equation, n_samples=args.eval_n_samples, batch_size=args.eval_batch_size)
        predictions[target] = prediction

    equation = build_equation(args)

    path = os.path.join("output", f"compare_{args.equation}_{'_'.join(targets)}")
    if not os.path.exists(path):
        os.mkdir(path)
    plot_losses(losses, show=False).savefig(os.path.join(path,"losses.png"))
    if args.equation == "ODE":
        lineplot(
            equation.x_ref, 
            equation.y_ref,
            predictions,
            xlabel="$x$", 
            ylabel="$u$", 
            show=False).savefig(os.path.join(path,"x_y_relation_2D.png"))
    elif args.equation == "Burgers":
        plot_y_distribution_2D(equation, predictions, align="col",show=False).savefig(os.path.join(path,"y_distribution_2D.png"))
    elif args.equation == "Darcy":
        plot_y_distribution_2D(equation, prediction, align="row", show=False).savefig(os.path.join(path,"y_distribution_2D.png"))
        with torch.no_grad():
            u = torch.linspace(-10, -4, 100)[:, None]
            k_exact= equation.K(u)
            k_pred = {}
            for key,model in zip(targets,models):
                k_pred[key] =  equation.correct_k(model.nn.stage[-1](u))
        
        lineplot(u, k_exact, k_pred, 
                 x_points= equation.y_u[:,0],
                 y_points= equation.k_u[:,0],
                 xlabel="$u$", ylabel="$K(u)$", show=False).savefig(os.path.join(path,"y_relation_2D.png"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None, required=True)
    parser.add_argument('-eq', '--equation', type=str, default='ODE', choices=['ODE', 'Burgers', 'Darcy'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('-on', '--device', type=str, default='cpu', choices=["cpu", "gpu"])
    parser.add_argument('--epoch', type=int, default=30000)
    parser.add_argument('--k1', type=int, default=1)
    parser.add_argument('--k2', type=int, default=5)
    parser.add_argument('--log_every_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-Nf', '--n_collosion', type=int, default=100)
    parser.add_argument('-Nu', '--n_boundary', type=int, default=100)
    parser.add_argument('-n', '--noise', type=float, default=0.05)
    parser.add_argument('--z_dim',  type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_layer_q', type=int, default=4)
    parser.add_argument('--n_layer_t', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=50)
    parser.add_argument('--n_hidden_q', type=int, default=50)
    parser.add_argument('--n_hidden_t', type=int, default=50)
    parser.add_argument('--nn', type=str, default="MLP", choices=["MLP","StackMLP"])
    parser.add_argument('--lambd', type=float, default=1.5)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--eval_n_samples', type=int, default=2000)
    parser.add_argument('-m','--model', type=str, default="uqpinn", choices=["uqpinn", "pinn"])
    parser.add_argument("-t","--task", default="main", choices=["main", "compare"])
    parser.add_argument("--targets", nargs="+", default=["pinn", "uqpinn"])
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
        args.n_layer = config.get('n_layer', args.n_layer)
        args.n_layer_q = config.get('n_layer_q', args.n_layer_q)
        args.n_layer_t = config.get('n_layer_t', args.n_layer_t)
        args.n_hidden = config.get('n_hidden', args.n_hidden)
        args.n_hidden_q = config.get('n_hidden_q', args.n_hidden_q)
        args.n_hidden_t = config.get('n_hidden_t', args.n_hidden_t)
        args.nn  = config.get('nn', args.nn)
        args.lambd = config.get('lambd', args.lambd)
        args.beta = config.get('beta', args.beta)
        args.lr = config.get('lr', args.lr)
        args.model = config.get('model', args.model)
        args.task = config.get('task', args.task)
        args.targets = config.get('targets', args.targets)
    

    {
        "main":main,
        "compare":compare
    }[args.task](args)
 
