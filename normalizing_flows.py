import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch
import numpy as np
from utils import create_dataset, read_dataset
from kl_estimator import one_sample_kl_estimator,two_sample_kl_estimator
from plot import plot_density, plot_dataset_mapping, plot_space_mapping, plot_training_loss, plot_jacobian_determinant


class Flow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2,128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,128)
        self.layer4 = nn.Linear(128,2)
    
    def forward(self,x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        x = self.layer4(x)
        return x

def loss_function(z_pred,J):
    """
    Loss function that computes the maximum likelihood-based loss.
    """
    dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((2,)),torch.eye(2))
    log_prob = dist.log_prob(z_pred)
    J_det = torch.abs(torch.linalg.det(J))
    loss = torch.mean(log_prob + torch.log(J_det))
    return -loss

def loss_function2(z_pred):
    """
    Loss function that computes the KL divergence loss based on a one-sample KL divergence estimator.
    """
    base_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((z_pred.shape[1])),torch.eye(z_pred.shape[1]))
    return one_sample_kl_estimator(z_pred,base_dist,z_pred.shape[0])

def loss_function3(z_pred,ref_sampling):
    """
    Loss function that computes the KL divergence loss based on a two-sample KL divergence estimator.
    """
    return two_sample_kl_estimator(z_pred,ref_sampling)

        

if __name__=="__main__":
    EXPERIMENT_NAME = "marc_2"
    TRAIN = False
    N_ITERATIONS = 1000
    torch.manual_seed(556358)

    for fp in [f"./figs/experiments/{EXPERIMENT_NAME}", f"./figs/experiments/{EXPERIMENT_NAME}/train",f"./logs/{EXPERIMENT_NAME}"]:
        isExist = os.path.exists(fp)
        if not isExist:
            os.makedirs(fp)


    # create_dataset(5000,"dataset2.h5")    
    x = read_dataset("dataset2.h5")

    if TRAIN:
        loss_list = []
        model = Flow()
        optim = torch.optim.Adam(model.parameters(),lr=0.001)

        for i in range(N_ITERATIONS):    #good training at least requires 200 to 500 steps
            torch.save(model.state_dict(),f"./logs/{EXPERIMENT_NAME}/model.pt")
            # plot_density(model,x,i,EXPERIMENT_NAME+"/train")    #comment out if training shouldln't be logged via images
            optim.zero_grad()
            z_pred = model(x)
            J = functorch.vmap(functorch.jacrev(model.forward))(x)
            loss = loss_function(z_pred,J)
            # loss = loss_function2(z_pred)
            # loss = loss_function3(z_pred,torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((2,)),torch.eye(2)).sample((10000,)))
            print(f"Iteration: {i+1}  -  Loss: {loss.detach().numpy()}")
            loss_list.append(loss.detach().numpy())
            np.savetxt(f"./logs/{EXPERIMENT_NAME}/loss.txt",loss_list)
            loss.backward()
            optim.step()

    else:
        model = Flow()
        model.load_state_dict(torch.load(f"./logs/{EXPERIMENT_NAME}/model.pt"))
        model.eval()

        plot_density(model,x,"final",EXPERIMENT_NAME,comparison=True)
        plot_dataset_mapping(model,x,EXPERIMENT_NAME,comparison=True)
        plot_space_mapping(model,x,EXPERIMENT_NAME)
        plot_training_loss(EXPERIMENT_NAME)
        plot_jacobian_determinant(model,x,EXPERIMENT_NAME)

    

   

