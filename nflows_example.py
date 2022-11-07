#%%
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet

from utils import read_dataset

#%%
x1, y1 = datasets.make_moons(128, noise=.1)

x2, y2 = datasets.make_moons(128, noise=.1)
x2 = x2 + 2; y2 = y2 + 2

x = np.concatenate((x1, x2), axis=0)
y = np.concatenate((y1, y2), axis=0)
plt.scatter(x[:, 0], x[:, 1], c=y);

#%%
num_layers = 5
base_dist = ConditionalDiagonalNormal(shape=[2], 
                                      context_encoder=nn.Linear(1, 4))

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4, 
                                                          context_features=1))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

#%%
num_iter = 5000
for i in range(num_iter):
    #x, y = datasets.make_moons(128, noise=.1)
    x1, y1 = datasets.make_moons(128, noise=.1)

    x2, y2 = datasets.make_moons(128, noise=.1)
    x2 = x2 + 2; y2 = y2 + 2

    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=y).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        fig, ax = plt.subplots(2, 2, figsize = (8,8))
        xline = torch.linspace(-2.0, 4.0, 200)
        yline = torch.linspace(-2.0, 4.0, 200)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid0 = flow.log_prob(xyinput, torch.zeros(40000, 1)).exp().reshape(200, 200)
            zgrid1 = flow.log_prob(xyinput, torch.ones(40000, 1)).exp().reshape(200, 200)
            zgrid2 = flow.log_prob(xyinput, torch.full([40000, 1], 2.0)).exp().reshape(200, 200)
            zgrid3 = flow.log_prob(xyinput, torch.full([40000, 1], 3.0)).exp().reshape(200, 200)

        ax[0, 0].set_aspect('equal')
        ax[0, 0].contourf(xgrid.numpy(), ygrid.numpy(), zgrid0.numpy(), 100)
        ax[0, 1].set_aspect('equal')
        ax[0, 1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1.numpy(), 100)
        ax[1, 0].set_aspect('equal')
        ax[1, 0].contourf(xgrid.numpy(), ygrid.numpy(), zgrid2.numpy(), 100)
        ax[1, 1].set_aspect('equal')
        ax[1, 1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid3.numpy(), 100)

        fig.suptitle('iteration {}'.format(i + 1))
        fig.tight_layout()
        plt.show()

# %%
import h5py
import numpy as np
with h5py.File("bicycle_dataset.h5", 'r') as f:
    position, time = np.array(f.get("position")), np.array(f.get("time"))

position = torch.Tensor(position).T
time = torch.Tensor(time)
# %%
plt.scatter(position[:, 0], position[:, 1], s = 1, c=time);
# %%
#%%
num_layers = 5
base_dist = ConditionalDiagonalNormal(shape=[2], 
                                      context_encoder=nn.Linear(1, 4))

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4, 
                                                          context_features=1))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

#%%
indices = torch.randperm(len(position))[:512]
position[indices]
time[indices]

#%%
num_iter = 1000
loss_arr = []
for i in range(num_iter):
    #x, y = datasets.make_moons(128, noise=.1)
    indices = torch.randperm(len(position))[:512]
    #x = torch.tensor(x, dtype=torch.float32)
    #y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    x = torch.tensor(position[indices], dtype=torch.float32)
    y = torch.tensor(time[indices], dtype=torch.float32).reshape(-1, 1)

    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=y).mean()
    loss_arr.append(loss)
    loss.backward()
    optimizer.step()
    
    '''
    if (i + 1) % 500 == 0:
        fig, ax = plt.subplots(1, 2)
        xline = torch.linspace(-1.5, 2.5, 100)
        yline = torch.linspace(-.75, 1.25, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid0 = flow.log_prob(xyinput, torch.zeros(10000, 1)).exp().reshape(100, 100)
            zgrid1 = flow.log_prob(xyinput, torch.ones(10000, 1)).exp().reshape(100, 100)

        ax[0].contourf(xgrid.numpy(), ygrid.numpy(), zgrid0.numpy())
        ax[1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()
    '''

# %%
import h5py
import numpy as np
with h5py.File("bicycle_dataset.h5", 'r') as f:
    position, time = np.array(f.get("position")), np.array(f.get("time"))

position = torch.Tensor(position).T
time = torch.Tensor(time)

position = position[time > 6.0]

# %%
plt.scatter(position[:, 0], position[:, 1], s = 1);
# %%
from nflows.distributions.normal import StandardNormal
num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    #transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4))
    transforms.append(ReversePermutation(features=2))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

#%%
num_iter = 5000
for i in range(num_iter):
    #x, y = datasets.make_moons(128, noise=.1)
    indices = torch.randperm(len(position))[:256]
    #x = torch.tensor(x, dtype=torch.float32)
    x = torch.tensor(position[indices], dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        xline = torch.linspace(-10.0, 90.0, 1000)
        yline = torch.linspace(-10.0, 90.0, 1000)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(1000, 1000)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy(), levels = 100)
        plt.title('iteration {}'.format(i + 1))
        plt.show()
# %%
