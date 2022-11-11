#%%
# Utility imports
import csv
import h5py
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

# PyTorch imports
import torch
from torch import nn
from torch import optim

# nflows imports
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.nn.nets import MLP
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from nflows.transforms import LULinear
from nflows.utils import torchutils

# %%
with h5py.File("bicycle_dataset_continuous.h5", 'r') as f:
    position, time = np.array(f.get("position")), np.array(f.get("time"))

mask = time>4
position = torch.Tensor(position).T
time = torch.Tensor(time)

# Test to remove deterministic start
position = position[mask]
time = time[mask]

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
    x = (x-torch.mean(x,dim=0))/torch.std(x,dim=0)
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

        new_xline = (xline*torch.std(position[indices][:,0],dim=0))+torch.mean(position[indices][:,0],dim=0)
        new_yline = (yline*torch.std(position[indices][:,1],dim=0))+torch.mean(position[indices][:,1],dim=0)
        new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

        plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), levels = 100, cmap="viridis")
        x = (x*torch.std(position[indices],dim=0))+torch.mean(position[indices],dim=0)
        plt.scatter(x[:,0], x[:,1], color = "k", s=1)
        plt.title('iteration {}'.format(i + 1))
        plt.show()

# %%
'''
with h5py.File("normalizing_flow_plot.h5", 'w') as f:
    f.create_dataset('x', data=new_xgrid.detach().numpy())
    f.create_dataset('y', data=new_ygrid.detach().numpy())
    f.create_dataset('z', data=zgrid.detach().numpy())
    f.create_dataset('position', data=x.detach().numpy())
    f.create_dataset('time', data=y.detach().numpy())
'''
# %%
# %%
with h5py.File("normalizing_flow_plot.h5", 'r') as f:
    zgrid = np.array(f.get("z"))
    x = np.array(f.get("x"))
    y = np.array(f.get("y"))
# %%
normalizing = 1/(0.2959711261379175*zgrid.sum())
# %%
my_context = torch.tensor([[13.0]])
data = flow.sample(200, my_context)
# %%
data = data.detach().squeeze()
plt.scatter(data[:,0], data[:,1], color = "k", s=1)
# %%

embedded_context = flow._embedding_net(my_context)
t = torch.linspace(0, 2*torch.pi, 100)
x = torch.sin(t)
y = torch.cos(t)
circle = torch.stack([x,y]).T
# %%
my_embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=100)
out, _ = flow._transform.inverse(circle, context = my_embedded_context)
out = out.detach()
plt.scatter(out[:,0], out[:,1], color = "k", s=1)
#%%
num_samples = 100
embedded_context = flow._embedding_net(my_context)

noise = flow._distribution.sample(num_samples, context=embedded_context)


if embedded_context is not None:
    # Merge the context dimension with sample dimension in order to apply the transform.
    noise = torchutils.merge_leading_dims(noise, num_dims=2)
    embedded_context = torchutils.repeat_rows(
        embedded_context, num_reps=num_samples
    )

#%%
samples, _ = flow._transform.inverse(noise, context=embedded_context)

if embedded_context is not None:
    # Split the context dimension from sample dimension.
    samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

# %%
