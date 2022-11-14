import torch
import torch.nn as nn
from kl_estimator import one_sample_kl_estimator,two_sample_kl_estimator
from utils import read_dataset, read_bicycle_dataset
import os
from plot import plot_space_mapping,plot_dataset_mapping,plot_density
import random
import matplotlib.pyplot as plt
import functorch
from metropolis_hastings import metropolis_hastings
import numpy as np
import h5py

EXPERIMENT_NAME = "vae_3"

for fp in [f"./figs/experiments/{EXPERIMENT_NAME}", f"./figs/experiments/{EXPERIMENT_NAME}/train",f"./logs/{EXPERIMENT_NAME}"]:
    isExist = os.path.exists(fp)
    if not isExist:
        os.makedirs(fp)

class VAE(nn.Module):
    def __init__(self,input_dim) -> None:
        super().__init__()
        self.enc_layer1 = nn.Linear(input_dim,64)
        self.enc_layer2a = nn.Linear(64,64)
        self.enc_layer2b = nn.Linear(64,64)
        self.enc_layer2c = nn.Linear(64,64)
        self.enc_layer2d = nn.Linear(64,64)
        self.enc_layer2e = nn.Linear(64,64)
        self.enc_layer2f = nn.Linear(64,64)
        self.enc_layer3 = nn.Linear(64,input_dim)
        self.dec_layer1 = nn.Linear(input_dim,64)
        self.dec_layer2a = nn.Linear(64,64)
        self.dec_layer2b = nn.Linear(64,64)
        self.dec_layer2c = nn.Linear(64,64)
        self.dec_layer2d = nn.Linear(64,64)
        self.dec_layer2e = nn.Linear(64,64)
        self.dec_layer2f = nn.Linear(64,64)
        self.dec_layer3 = nn.Linear(64,input_dim)
        # self.latent_mu = nn.Parameter(-0.5 * torch.ones((2,)))
        # self.latent_log_std = nn.Parameter(torch.zeros((2,)))
          
    def encode(self,x):
        z = self.enc_layer1(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2a(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2b(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2c(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2d(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2e(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2f(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer3(z)
        
        return z

    def decode(self,z):
        x = self.enc_layer1(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2a(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2b(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2c(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2d(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2e(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2f(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer3(x)
    
        return x
    
    def forward(self,x):
        return self.encode(x)

input_dim = 2
epochs = 300
lr = 0.001
model = VAE(input_dim)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

x_complete_all,t = read_bicycle_dataset("bicycle_dataset_continuous.h5")
x_complete_unscaled = x_complete_all[t==13,:]

mu = torch.mean(x_complete_unscaled,dim=0)
std = torch.std(x_complete_unscaled,dim=0)
x_complete = (x_complete_unscaled-mu)/std


# for i in range(epochs):
#     # if i ==150:
#     #     lr_scale = 0.5
#     #     print(f"Changing Learning Rate to {lr*lr_scale}")
#     #     optimizer.param_groups[0]["lr"] = lr*lr_scale
#     #     print(f"Double check new learning rate is {optimizer.param_groups[0]['lr']}")
#     # if i ==100:
#     #     lr_scale = 0.5
#     #     print(f"Changing Learning Rate to {lr*lr_scale}")
#     #     optimizer.param_groups[0]["lr"] = lr*lr_scale
#     #     print(f"Double check new learning rate is {optimizer.param_groups[0]['lr']}")
#     indices = torch.tensor(random.sample(range(x_complete.shape[0]), 5000))
#     indices = torch.tensor(indices)
#     x = x_complete[indices]
#     optimizer.zero_grad()
#     z_predict = model.encode(x)
#     x_predict = model.decode(z_predict)
#     latent_target_dist = torch.distributions.multivariate_normal.MultivariateNormal(model.latent_mu,torch.diag((model.latent_log_std.exp())**2))
#     latent_loss = one_sample_kl_estimator(z_predict,latent_target_dist,5000)
#     reconstruction_loss = torch.linalg.norm(x-x_predict,dim=1).mean()
#     # reconstruction_loss = torch.nn.functional.mse_loss(x_predict,x)
#     # reconstruction_loss = two_sample_kl_estimator(x,x_predict)
#     loss = 5 * latent_loss + reconstruction_loss
#     print(f"Epoch {i}/{epochs} - Loss: {loss.detach().item()}")
#     loss.backward()
#     optimizer.step()

# # save model
# torch.save(model.state_dict(), "vae_model3.pth")

#load model
model.load_state_dict(torch.load("vae_model2.pth"))


def get_prob(x,model):
    ref_dist = torch.distributions.MultivariateNormal(torch.zeros((2,)),torch.eye(2))
    preds = model(x.reshape((-1,2)))
    J = functorch.vmap(functorch.jacrev(model.forward))(x.reshape((-1,2)))
    probs = torch.exp(ref_dist.log_prob(preds)) * torch.abs(torch.linalg.det(J)) /0.7
    return probs.detach()

num_samples = 20000
samples = metropolis_hastings(get_prob,model,num_samples=num_samples,num_start=5000)
samples = torch.stack(samples)
probs = get_prob(samples,model)
idx = torch.argsort(probs,descending=True)
cutoff_1 = int(num_samples*0.6827)
cutoff_2 = int(num_samples*0.9545)
cutoff_3 = int(num_samples*0.9973)
prob_crit_1 = probs[idx[cutoff_1]]
# prob_crit_1 = 0.1438
prob_crit_2 = probs[idx[cutoff_2]]
# prob_crit_2 = 0.0198
prob_crit_3 = probs[idx[cutoff_3]]
# prob_crit_3 = 0.0053

with h5py.File("samples_vae.h5", 'w') as f:
    f.create_dataset('samples', data=(samples[torch.randperm(20000)[:1000],:]*std)+mu)

print("stop")
# X,Y = torch.meshgrid([torch.linspace(-10,10,1000),torch.linspace(-10,10,1000)])
# grid_points = torch.stack((X,Y),dim=-1).reshape((-1,2))
# grid_probs = get_prob(grid_points,model).reshape((1000,1000))
# cn = plt.contour(X*std[0]+ mu[0],Y*std[1]+mu[1],grid_probs,levels=[prob_crit_3,prob_crit_2,prob_crit_1])
# plt.scatter(x_complete_all[:15000,0],x_complete_all[:15000,1],s=1,color=(0.5,0.5,0.5))
# plt.scatter(x_complete_unscaled[:108,0],x_complete_unscaled[:108,1],s=1,color=(1,0,0))
# plt.savefig("test.png",dpi=600)
# # plt.contourf(X,Y,grid_probs)
# # plt.scatter(samples[:,0],samples[:,1],s=1)
# plt.show()
# print("stop")

# contours = []
# # for each contour line
# for cc in cn.collections:
#     paths = []
#     # for each separate section of the contour line
#     for pp in cc.get_paths():
#         xy = []
#         # for each segment of that section
#         for vv in pp.iter_segments():
#             xy.append(vv[0])
#         paths.append(np.vstack(xy))
#     contours.append(paths)

# with h5py.File("flow_level_sets_vae.h5", 'w') as f:
#     f.create_dataset('x68', data=contours[0][0][:,0])
#     f.create_dataset('y68', data=contours[0][0][:,1])
#     f.create_dataset('x95', data=contours[1][0][:,0])
#     f.create_dataset('y95', data=contours[1][0][:,1])
#     f.create_dataset('x995', data=contours[2][0][:,0])
#     f.create_dataset('y995', data=contours[2][0][:,1])



# plt.show()
# print("stop")

# plot_density(model,x_complete,6767,EXPERIMENT_NAME)
# plot_space_mapping(model,x_complete,EXPERIMENT_NAME)
# plot_dataset_mapping(model,x_complete,EXPERIMENT_NAME,comparison=True)

print("stop")
