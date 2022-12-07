#%%
# Utility imports
import copy
import h5py
import matplotlib.pyplot as plt
import numpy as np

from plot import make_gif
from scipy.linalg import sqrtm
from tqdm import tqdm

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
# Read in the dataset
with h5py.File("bicycle_dataset_bimodal.h5", 'r') as f:
    position, time = np.array(f.get("position")), np.array(f.get("time"))

position = torch.Tensor(position).T
time = torch.Tensor(time)

# Generate noisy observationa
obs_sigma = 1.0
obs = copy.deepcopy(position)

torch.manual_seed(3456271)
obs += torch.distributions.Normal(0.0,obs_sigma).sample(obs.shape)

# Standardize the data
pos_mean = torch.mean(obs, dim=0)
pos_std = torch.std(obs, dim=0)
position = (position - pos_mean)/pos_std
obs = (obs - pos_mean)/pos_std

# Visualize the dataset
plt.scatter(position[0:2000, 0], position[0:2000, 1], s = 1, c=time[0:2000]);
plt.gca().set_aspect('equal'); plt.grid()

#%%
# Find the position, time, and observation data associated with each trajectory
end_idxs = np.where(np.diff(time) <= 0.0)[0]
pos_seqs = np.array_split(position, end_idxs + 1, axis=0)
time_seqs = np.array_split(time, end_idxs+1)
obs_seqs = np.array_split(obs, end_idxs + 1, axis=0)

# Find lengths of individual trajectories
seq_lens = [s.shape[0] for s in pos_seqs]

# Create sequence data
min_seq_len = 5
feats = []; targets = []

for (i, o) in enumerate(obs_seqs):
    feats += [torch.cat((o[0:j, :], time_seqs[i][0:j].unsqueeze(0).T), dim=1) 
                for j in range(min_seq_len, o.shape[0])]
    targets += [pos_seqs[i][k, :] for k in range(min_seq_len, o.shape[0])]

inputs = torch.vstack(targets).to(torch.float32)
feat_lens = torch.Tensor([f.shape[0] for f in feats])

#%%
# Pad features
batch_first = True
pad_val = torch.nan
feats_padded = torch.nn.utils.rnn.pad_sequence(
    feats, batch_first=batch_first, padding_value=pad_val).to(torch.float32)

#%%
# GRU Module
class GRUContextEncoder(nn.Module):
    """ GRU Context Encoder

    Args:
        input_features (int): Number of input features in input sequences.
        hidden_size (int): Number of hidden units.            
        out_features (int): Number of output features.
        seq_len (int): Input sequence lengths.
        layers (int): GRU layers.

    """
    def __init__(self, input_features, hidden_size, out_features, seq_len=31, layers=1):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.layers = layers
        self.seq_len = seq_len

        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=self.layers
        )

        self.linear = nn.Linear(in_features=hidden_size, out_features=out_features)

    def forward(self, x):
        masked_ctx = []
        for ctx in x:
            mask = ~torch.isnan(ctx)
            masked_ctx.append(ctx[mask].reshape(-1, 3))
        padded = torch.nn.utils.rnn.pack_sequence(masked_ctx, enforce_sorted=False)
        gru_out, hn = self.gru(padded)
        out = self.linear(hn.squeeze()).squeeze()
        return out

#%%
# Define the context encoder
input_size = 3
hidden_size = 4
embed_size = 4
gru_encoder = GRUContextEncoder(input_size, hidden_size, embed_size)

#%%
# Define the latent base distribution
base_dist = ConditionalDiagonalNormal(
                shape=[2], context_encoder=nn.Sequential(
                    nn.Linear(4,32), nn.ReLU(), nn.Linear(32,4)))
# Define the flow transform
num_layers = 10
transform_list = []
for _ in range(num_layers):
    transform_list.append(RandomPermutation(features=2))
    transform_list.append(LULinear(2, identity_init=True)),
    transform_list.append(MaskedAffineAutoregressiveTransform(
        features=2, 
        hidden_features=4,
        context_features=4,
        use_batch_norm=False))
transform = CompositeTransform(transform_list)
flow_gru = Flow(transform, base_dist, embedding_net=gru_encoder)

# Define the optimizer
optimizer = optim.Adam(flow_gru.parameters(), lr=1e-3)

#%%
# Train the flow
num_iter = 3000
loss_arr = []
for i in range(num_iter):
    indices = torch.randperm(feats_padded.shape[0])[:2048].to(int)
    x = inputs[indices, :]
    ctx = feats_padded[indices, :, :]

    # Compute the loss
    optimizer.zero_grad()
    loss = -flow_gru.log_prob(inputs=x, context=ctx).mean()
    loss_arr.append(loss.detach())
    loss.backward()
    optimizer.step()
    
    # Plot candidate PDF
    if (i + 1) % 500 == 0:
        xline = torch.linspace(-4.0, 4.0, 200)
        yline = torch.linspace(-4.0, 4.0, 200)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
        
        with torch.no_grad():
            ctx_test = feats_padded[100, :, :]
            zgrid = flow_gru.log_prob(
                xyinput, ctx_test.repeat(40000, 1, 1)).exp().reshape(200, 200)

        new_xline = (xline*pos_std[0])+pos_mean[0]
        new_yline = (yline*pos_std[1])+pos_mean[1]
        new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

        plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), 
            levels=500, cmap="viridis")
        x = (x*pos_std)+pos_mean

        m = targets[100] * pos_std + pos_mean
        m = m.numpy()
        plt.scatter(m[0], m[1], color='r', s=20)
        plt.title('iteration {}'.format(i + 1))
        plt.show()

# Save the loss data
np.savetxt("figs/loss/gru_loss.csv", loss_arr)

#%% 
# Evaluate the log likelihood of the model
flow_gru.eval()
feats = feats_padded
idx = torch.arange(feats.shape[0])
idx_list = torch.split(idx,8192)
sum_log_probs = 0
for ix in tqdm(idx_list):
    x = inputs[ix, :]
    ctx = feats[ix, :, :]
    with torch.inference_mode():
        log_prob = flow_gru.log_prob(inputs=x, context=ctx).detach().sum().item()
    sum_log_probs += log_prob

mean_log_probs = sum_log_probs/idx.shape[0]
print(f"Model Score: {mean_log_probs}")
flow_gru.train()
print("stop")

#%%
# Make a gif
seq_idx = 6
test_obs = obs_seqs[seq_idx]
test_pos = pos_seqs[seq_idx]

xline = torch.linspace(-4.0, 4.0, 200)
yline = torch.linspace(-4.0, 4.0, 200)
xgrid, ygrid = torch.meshgrid(xline, yline)
xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
new_xline = (xline*pos_std[0])+pos_mean[0]
new_yline = (yline*pos_std[1])+pos_mean[1]
new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

ctx_test = pad_val * torch.ones(31, 3, dtype=torch.float32)

for i in range(min_seq_len, seq_lens[seq_idx]):    
    with torch.no_grad():
        ctx_test[0:i, 0:2] = test_obs[0:i, :]
        ctx_test[0:i, 2] = time_seqs[seq_idx][0:i]
        zgrid = flow_gru.log_prob(
            xyinput, ctx_test.repeat(40000, 1, 1)).exp().reshape(200, 200)

    plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), 
        levels = 500, cmap="viridis")

    state_i = (test_pos[:i, :] * pos_std + pos_mean).numpy()
    obs_i = (test_obs[:i, :] * pos_std + pos_mean).numpy()
    plt.scatter(obs_i[:, 0], obs_i[:, 1], color='k', s=3)
    plt.plot(state_i[:, 0], state_i[:, 1], '--', color='r')
    plt.title('t = {}'.format(i + 1))
    plt.xlim((0, 100))
    plt.ylim((-10, 60))
    plt.savefig("./figs/experiments/gru/tmp/tmp_{:03d}.png".format(i))
    plt.show()

make_gif("./figs/experiments/gru/tmp/", "./figs/experiments/gru/seq6.gif", delete_frames=True)

#%%
# Make a contour plot
seq_idx = 6
test_obs = obs_seqs[seq_idx]
test_pos = pos_seqs[seq_idx]

xline = torch.linspace(-4.0, 4.0, 200)
yline = torch.linspace(-4.0, 4.0, 200)
xgrid, ygrid = torch.meshgrid(xline, yline)
xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
new_xline = (xline*pos_std[0])+pos_mean[0]
new_yline = (yline*pos_std[1])+pos_mean[1]
new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

ctx_test = pad_val * torch.ones(31, 3, dtype=torch.float32)

i = 10
with torch.no_grad():
    ctx_test[0:i, 0:2] = test_obs[0:i, :]
    ctx_test[0:i, 2] = time_seqs[seq_idx][0:i]
    zgrid = flow_gru.log_prob(
        xyinput, ctx_test.repeat(40000, 1, 1)).exp().reshape(200, 200)

plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), 
    levels = 500, cmap="viridis")

state_i = (test_pos[:i, :] * pos_std + pos_mean).numpy()
obs_i = (test_obs[:i, :] * pos_std + pos_mean).numpy()
plt.scatter(obs_i[:, 0], obs_i[:, 1], color='k', s=3)
plt.plot(state_i[:, 0], state_i[:, 1], '--', color='r')
plt.title('t = {}'.format(i + 1))
plt.xlim((0, 100))
plt.ylim((-10, 60))
plt.savefig("./figs/experiments/gru/gru_pdf.png")
plt.show()

# %%
# Extract level sets

# Define the context (time that we're conditioning on)
seq_idx = 6
test_obs = obs_seqs[seq_idx]
test_pos = pos_seqs[seq_idx]

xline = torch.linspace(-4.0, 4.0, 200)
yline = torch.linspace(-4.0, 4.0, 200)
xgrid, ygrid = torch.meshgrid(xline, yline)
xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
new_xline = (xline*pos_std[0])+pos_mean[0]
new_yline = (yline*pos_std[1])+pos_mean[1]
new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

ctx_test = pad_val * torch.ones(31, 3, dtype=torch.float32)

i = 10
with torch.no_grad():
    ctx_test[0:i, 0:2] = test_obs[0:i, :]
    ctx_test[0:i, 2] = time_seqs[seq_idx][0:i]

ctx_test = ctx_test.unsqueeze(0)

embedded_context = flow_gru._embedding_net(ctx_test)
rep_embedded_context = torchutils.repeat_rows(embedded_context.unsqueeze(0), num_reps=400)

params = flow_gru._distribution._compute_params(embedded_context.unsqueeze(0))

log_stds = params[1].detach().squeeze()
mu = params[0].detach().squeeze().numpy()

Sigma = torch.diag((log_stds).exp()**2).numpy()

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

ex1, ey1 = error_ellipse(mu, Sigma, p1)
ex2, ey2 = error_ellipse(mu, Sigma, p2)
ex3, ey3 = error_ellipse(mu, Sigma, p3)

circle1 = torch.stack([ex1, ey1]).T
circle2 = torch.stack([ex2, ey2]).T
circle3 = torch.stack([ex3, ey3]).T

region1, _ = flow_gru._transform.inverse(circle1, context = rep_embedded_context)
region2, _ = flow_gru._transform.inverse(circle2, context = rep_embedded_context)
region3, _ = flow_gru._transform.inverse(circle3, context = rep_embedded_context)

region1 = torchutils.split_leading_dim(region1, shape=[-1, 400])
region2 = torchutils.split_leading_dim(region2, shape=[-1, 400])
region3 = torchutils.split_leading_dim(region3, shape=[-1, 400])

region1 = (region1.detach()*pos_std)+pos_mean
region2 = (region2.detach()*pos_std)+pos_mean
region3 = (region3.detach()*pos_std)+pos_mean

fig, axs = plt.subplots(1, 2)
axs[0].plot(ex1, ey1); axs[0].plot(ex2, ey2); axs[0].plot(ex3, ey3); 
axs[0].set_aspect('equal'); axs[0].grid(True)
axs[0].set_title("Latent Space")

axs[1].plot(region1.squeeze()[:,0], region1.squeeze()[:,1]); 
axs[1].plot(region2.squeeze()[:,0], region2.squeeze()[:,1]); 
axs[1].plot(region3.squeeze()[:,0], region3.squeeze()[:,1]); 

axs[1].set_aspect('equal'); axs[1].grid(True)
axs[1].set_title("Driving Scene")
# %%
with h5py.File("flow_level_sets_gru.h5", 'w') as f:
    f.create_dataset('x68', data=region1.squeeze()[:,0])
    f.create_dataset('y68', data=region1.squeeze()[:,1])
    f.create_dataset('x95', data=region2.squeeze()[:,0])
    f.create_dataset('y95', data=region2.squeeze()[:,1])
    f.create_dataset('x995', data=region3.squeeze()[:,0])
    f.create_dataset('y995', data=region3.squeeze()[:,1])
# %%
