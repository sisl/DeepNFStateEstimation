#%%
# Utility imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

# PyTorch imports
import torch
from torch import nn
from torch import optim

# nflows imports
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms import LULinear
from nflows.utils import torchutils

# %%
with h5py.File("bicycle_dataset_continuous.h5", 'r') as f:
    position, time = np.array(f.get("position")), np.array(f.get("time"))

position = torch.Tensor(position).T
time = torch.Tensor(time)

pos_mean = torch.mean(position, dim=0)
pos_std = torch.std(position, dim=0)
position = (position - pos_mean)/pos_std

# Test to remove deterministic start
#mask = time>2
#position = position[mask]
#time = time[mask]

plt.scatter(position[0:2000, 0], position[0:2000, 1], s = 1, c=time[0:2000]);
plt.gca().set_aspect('equal'); plt.grid()
#%%
num_layers = 10
base_dist = ConditionalDiagonalNormal(shape=[2], 
#                                      context_encoder=MLP([1], [4], hidden_sizes=[32,32]))
                                        context_encoder=nn.Sequential(nn.Linear(1,32), nn.ReLU(), nn.Linear(32,4)))
transform_list = []
for _ in range(num_layers):
    transform_list.append(RandomPermutation(features=2))
    transform_list.append(LULinear(2, identity_init=True)),
    transform_list.append(MaskedAffineAutoregressiveTransform(
        features=2, 
        hidden_features=4,
        context_features=1,
        use_batch_norm=False))
transform = CompositeTransform(transform_list)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())


#%%
num_iter = 3000
loss_arr = []
for i in range(num_iter):
    indices = torch.randperm(len(position))[:2048]
    x = torch.tensor(position[indices], dtype=torch.float32)
    y = torch.tensor(time[indices], dtype=torch.float32).reshape(-1, 1)

    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=y).mean()
    loss_arr.append(loss.detach())
    loss.backward()
    optimizer.step()
    
    if (i < 10) or (i + 1) % 500 == 0:
        xline = torch.linspace(-4.0, 4.0, 200)
        yline = torch.linspace(-4.0, 4.0, 200)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
        
        with torch.no_grad():
            zgrid = flow.log_prob(xyinput, torch.full([40000, 1], 13.0)).exp().reshape(200, 200)

        new_xline = (xline*pos_std[0])+pos_mean[0]
        new_yline = (yline*pos_std[1])+pos_mean[1]
        new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

        plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), levels = 100, cmap="viridis")
        x = (x*pos_std)+pos_mean

        y_mask = (y==13.0).squeeze()
        red_x = x[y_mask, :]

        plt.scatter(x[:,0], x[:,1], color = "k", s=1)
        plt.scatter(red_x[:,0], red_x[:,1], color = "r", s=3)
        plt.title('iteration {}'.format(i + 1))
        plt.show()

#%%
# Define the context (time that we're conditioning on)
context = torch.tensor([[13.0]])
embedded_context = flow._embedding_net(context)
rep_embedded_context = torchutils.repeat_rows(embedded_context, num_reps=400)

params = flow._distribution._compute_params(embedded_context)
log_stds = params[1].detach().squeeze()
mu = params[0].detach().squeeze().numpy()

Sigma = torch.diag((log_stds).exp()**2).numpy()


#mu = np.array([1.0, -6.0]); Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
p1 = 0.68; p2 = 0.95; p3 = 0.995

def error_ellipse(mu, Sigma, prob):
    r = np.sqrt(-2*np.log(1-prob))
    angles = torch.linspace(0, 2*np.pi, 400)
    cx = [r*np.cos(a) for a in angles];
    cy = [r*np.sin(a) for a in angles];

    ellipse = np.matmul(sqrtm(Sigma), (np.vstack([cx, cy]))) + np.reshape(mu, (2,1))
    ellipse = torch.Tensor(ellipse)

    ex = ellipse[0,:]; ey = ellipse[1,:]

    return ex, ey

#%%
ex1, ey1 = error_ellipse(mu, Sigma, p1)
ex2, ey2 = error_ellipse(mu, Sigma, p2)
ex3, ey3 = error_ellipse(mu, Sigma, p3)

#%%
circle1 = torch.stack([ex1, ey1]).T
circle2 = torch.stack([ex2, ey2]).T
circle3 = torch.stack([ex3, ey3]).T

region1, _ = flow._transform.inverse(circle1, context = rep_embedded_context)
region2, _ = flow._transform.inverse(circle2, context = rep_embedded_context)
region3, _ = flow._transform.inverse(circle3, context = rep_embedded_context)

region1 = torchutils.split_leading_dim(region1, shape=[-1, 400])
region2 = torchutils.split_leading_dim(region2, shape=[-1, 400])
region3 = torchutils.split_leading_dim(region3, shape=[-1, 400])

region1 = (region1.detach()*pos_std)+pos_mean
region2 = (region2.detach()*pos_std)+pos_mean
region3 = (region3.detach()*pos_std)+pos_mean

# %%
fig, axs = plt.subplots(1, 2)
axs[0].plot(ex1, ey1); axs[0].plot(ex2, ey2); axs[0].plot(ex3, ey3); 
axs[0].set_aspect('equal'); axs[0].grid(True)
axs[0].set_title("Latent Space")

axs[1].plot(region1.squeeze()[:,0], region1.squeeze()[:,1]); 
axs[1].plot(region2.squeeze()[:,0], region2.squeeze()[:,1]); 
axs[1].plot(region3.squeeze()[:,0], region3.squeeze()[:,1]); 

axs[1].set_aspect('equal'); axs[1].grid(True)
axs[1].set_title("Driving Scene")


#%%
sample_data = position[time==13, :][1:200]
sample_data = sample_data*pos_std + pos_mean
plt.scatter(sample_data[:,0], sample_data[:,1], s = 1)

#%%
context = torch.tensor([[13.0]])
embedded_context = flow._embedding_net(context)
outputs, _ = flow._transform(sample_data, context = embedded_context)
out = outputs.detach()
plt.scatter(out[:, 0], out[:, 1], s = 1);



#%%
samples = flow.sample(1000, context = context)
samples = samples.detach().squeeze()

samples = samples*pos_std + pos_mean
plt.scatter(samples[:, 0], samples[:, 1], s = 1);
plt.scatter(sample_data[:, 0], sample_data[:, 1], s = 1, c = "r");
plt.gca().set_aspect('equal'); plt.grid()

#%%
with h5py.File("samples_nflow.h5", 'w') as f:
    f.create_dataset('samples', data=samples)

#%%
params = flow._distribution._compute_params(embedded_context)
log_stds = params[1].detach().squeeze()
means = params[0].detach().squeeze()

Cov_Matrix = torch.diag((log_stds).exp()**2)

p = 0.997
ex, ey = error_ellipse(means.numpy(), Cov_Matrix.numpy(), p)

#%%
noise = flow._distribution.sample(5000, context=embedded_context)
noise = noise.detach().squeeze()
plt.scatter(noise[:, 0], noise[:, 1], s = 1);
plt.plot(ex, ey)

# %%
'''
with h5py.File("flow_level_sets.h5", 'w') as f:
    f.create_dataset('x68', data=region1.squeeze()[:,0])
    f.create_dataset('y68', data=region1.squeeze()[:,1])
    f.create_dataset('x95', data=region2.squeeze()[:,0])
    f.create_dataset('y95', data=region2.squeeze()[:,1])
    f.create_dataset('x995', data=region3.squeeze()[:,0])
    f.create_dataset('y995', data=region3.squeeze()[:,1])
# %%
# UKF samples
mu_ukf = np.array([65.1842,  31.667])
Sigma_ukf = np.array([[27.1505,  -54.2592], [-54.2592,  115.653]])
samples_ukf = np.random.multivariate_normal(mu_ukf, Sigma_ukf, 1000)
plt.scatter(samples_ukf[:, 0], samples_ukf[:, 1], s = 1);
# %%
with h5py.File("samples_ukf.h5", 'w') as f:
    f.create_dataset('samples', data=samples_ukf)

# %%
with h5py.File("bicycle_dataset_continuous.h5", 'r') as f:
    position, time = np.array(f.get("position")), np.array(f.get("time"))

position = torch.Tensor(position).T
time = torch.Tensor(time)

context_time = time==13.0

sample_truth = position[context_time, :][:1000]
# %%
with h5py.File("samples_truth.h5", 'w') as f:
    f.create_dataset('samples', data=sample_truth)

# %%
'''