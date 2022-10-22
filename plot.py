import torch
import matplotlib.pyplot as plt
import numpy as np
import functorch
from utils import reference, color_grid

def plot_density(model, x, i, base_name):
    model.eval()

    ref_dist = torch.distributions.MultivariateNormal(torch.zeros((x.shape[1])),torch.eye(x.shape[1]))

    x1 = torch.linspace(-2,5,100)
    x2 = torch.linspace(-1,7,100)
    X = torch.meshgrid((x1,x2))
    X = torch.stack(X,dim=-1)
    X_flat = torch.reshape(X,(-1,2))
    preds = model(X_flat)
    J = functorch.vmap(functorch.jacrev(model.forward))(X_flat)
    probs = torch.exp(ref_dist.log_prob(preds)) * torch.abs(torch.linalg.det(J))
    probs = probs.reshape(X[:,:,-1].shape)
    reference_probs = reference(X_flat).reshape(X[:,:,-1].shape)    #these are the ground truth likelihoods

    #plot density
    plt.rcParams['text.usetex'] = True
    plt.contourf(X[:,:,0],X[:,:,1],probs.detach().numpy(),levels=50)
    plt.scatter(x[:,0],x[:,1],s=1,color=(0,0,0))
    plt.xlim(-2,5)
    plt.ylim(-1,7)
    plt.savefig(f"./figs/experiments/{base_name}/density_{i}.png",dpi=600)
    plt.clf()
    model.train()

def plot_dataset_mapping(model,x,basename):
    model.eval()

    z_pred = model(x)
    plt.rcParams['text.usetex'] = True
    fig,axs = plt.subplots(1,1,figsize=(5,5))
    axs.scatter(z_pred[:,0].detach().numpy(),z_pred[:,1].detach().numpy(),s=1)
    axs.set_xlabel(r"$z_1$")
    axs.set_ylabel(r"$z_2$")
    # axs[1].scatter(z_pred[:,1].detach().numpy(),z_pred[:,2].detach().numpy(),s=1)
    # axs[1].set_xlabel(r"$z_2$")
    # axs[1].set_xlabel(r"$z_3$")
    # axs[2].scatter(z_pred[:,0].detach().numpy(),z_pred[:,2].detach().numpy(),s=1)
    # axs[2].set_xlabel(r"$z_1$")
    # axs[2].set_xlabel(r"$z_3$")
    fig.suptitle(r"Dataset mapped from $\mathbf{X}$ to $\mathbf{Z}$")
    fig.tight_layout()
    fig.savefig(f"./figs/experiments/{basename}/dataset_mapping.png",dpi=600)
    fig.show()

def plot_space_mapping(model,x,basename):
    x1_min = x[:,0].min()
    x1_max = x[:,0].max()
    x2_min = x[:,1].min()
    x2_max = x[:,1].max()

    x_x, colors = color_grid(50,50,x1_min,x1_max,x2_min,x2_max)
    x_z = model(torch.Tensor(x_x)).detach().numpy()

    plt.rcParams['text.usetex'] = True
    plt.style.use('dark_background')
    fig,axs = plt.subplots(1,2,figsize=(10, 5))
    axs[0].scatter(x_x[:,0],x_x[:,1],s=3,c=colors)
    axs[0].set_xlabel(r"$x_1$")
    axs[0].set_ylabel(r"$x_2$")

    axs[1].scatter(x_z[:,0],x_z[:,1],s=3,c=colors)
    axs[1].set_xlabel(r"$z_1$")
    axs[1].set_ylabel(r"$z_2$")

    # axs[2].scatter(x_z[:,1],x_z[:,2],s=3,c=colors)
    # axs[2].set_xlabel(r"$z_2$")
    # axs[2].set_ylabel(r"$z_3$")

    # axs[3].scatter(x_z[:,0],x_z[:,2],s=3,c=colors)
    # axs[3].set_xlabel(r"$z_1$")
    # axs[3].set_ylabel(r"$z_3$")

    fig.suptitle(r"Mapping between $\mathbf{X}$ and $\mathbf{Z}$")
    fig.tight_layout()
    fig.savefig(f"./figs/experiments/{basename}/space_mapping.png",dpi=600)
    fig.show()
    plt.style.use('default')

def plot_training_loss(basename):
    l = np.loadtxt(f"./logs/{basename}/loss.txt")
    plt.rcParams['text.usetex'] = True
    fig,axs = plt.subplots(1,1,figsize=(10,5))
    axs.plot(l)
    axs.grid(which="both")
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    fig.suptitle("Loss During Training")
    fig.tight_layout()
    fig.savefig(f"./figs/experiments/{basename}/train_loss.png",dpi=600)
    fig.show()