#%%
# Utility imports
import h5py
import matplotlib.pyplot as plt
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

# %%
with h5py.File("bicycle_dataset_discrete.h5", 'r') as f:
    position, time = np.array(f.get("position")), np.array(f.get("time"))

mask = time>2
position = torch.Tensor(position).T
time = torch.IntTensor(time)

# Test to remove deterministic start
position = position[mask]
time = time[mask]

plt.scatter(position[0:2000, 0], position[0:2000, 1], s = 1, c=time[0:2000]);
#%%
num_layers = 10
base_dist = ConditionalDiagonalNormal(shape=[2], 
                                      context_encoder=MLP([1], [4], hidden_sizes=[32,32]))
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
num_iter = 2500
loss_arr = []
for i in range(num_iter):
    indices = torch.randperm(len(position))[:1024]
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
            zgrid = flow.log_prob(xyinput, torch.full([40000, 1], 14.0)).exp().reshape(200, 200)

        new_xline = (xline*torch.std(position[indices][:,0],dim=0))+torch.mean(position[indices][:,0],dim=0)
        new_yline = (yline*torch.std(position[indices][:,1],dim=0))+torch.mean(position[indices][:,1],dim=0)
        new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

        plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), levels = 100)
        x = (x*torch.std(position[indices],dim=0))+torch.mean(position[indices],dim=0)
        plt.scatter(x[:,0], x[:,1], color = "k", s=1)
        plt.title('iteration {}'.format(i + 1))
        plt.show()

# %%
