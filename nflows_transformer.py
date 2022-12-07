#%%
# Utility imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import copy
from plot import make_gif
from transformer_utils import TransformerEmbedding
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
with h5py.File("bicycle_dataset_bimodal.h5", 'r') as f:
    position, time = np.array(f.get("position")), np.array(f.get("time"))

position = torch.Tensor(position).T
time = torch.Tensor(time)

obs_sigma = 1.0
obs = copy.deepcopy(position)
obs += np.random.normal(0.0, obs_sigma, obs.shape)
#obs = (position - pos_mean)/pos_std

pos_mean = torch.mean(obs, dim=0)
pos_std = torch.std(obs, dim=0)
position = (position - pos_mean)/pos_std
obs = (obs - pos_mean)/pos_std

plt.scatter(position[0:2000, 0], position[0:2000, 1], s = 1, c=time[0:2000]);
plt.gca().set_aspect('equal'); plt.grid()

#%%
batch_first = True

#length of time sequence
end_idxs = np.where(np.diff(time) <= 0.0)[0]
pos_seqs = np.array_split(position, end_idxs + 1, axis=0)
time_seqs = np.array_split(time, end_idxs+1)
seq_lens = [s.shape[0] for s in pos_seqs]

obs_seqs = np.array_split(obs, end_idxs + 1, axis=0)

#create sequence data
min_seq_len = 5
feats = []
targets = []

for (i, o) in tqdm(enumerate(obs_seqs)):
    #feats += [o[0:j, :] for j in range(min_seq_len, o.shape[0])]
    # feats += [torch.cat((o[0:j, :], time_seqs[i][0:j].unsqueeze(0).T), dim=1) for j in range(min_seq_len, o.shape[0])]
    feats += [torch.cat((o[j-min_seq_len:j, :], time_seqs[i][j-min_seq_len:j].unsqueeze(0).T), dim=1).to(torch.float32) for j in range(min_seq_len, o.shape[0])]
    targets += [pos_seqs[i][k, :] for k in range(min_seq_len, o.shape[0])]

feat_lens = torch.Tensor([f.shape[0] for f in feats])


# pad/pack features
# pad_val = -1.0
# feats_padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=batch_first, padding_value=pad_val).to(torch.float32)
# feats_packed = torch.nn.utils.rnn.pack_padded_sequence(feats_padded, feat_lens, batch_first=batch_first, enforce_sorted=False)

feats = torch.stack(feats,dim=0)
inputs = torch.vstack(targets).to(torch.float32)



# Transformer Module
class TransformerContextEncoder(nn.Module):

    def __init__(self,num_hidden,num_output):
        super().__init__()
        self.te = TransformerEmbedding(num_hidden)
        self.linear = nn.Linear(num_hidden,num_output)

    def forward(self, x):
        x = x.permute(1,0,2)
        pred = self.te(x).squeeze()
        output = self.linear(pred)
        return output



input_size = 3
hidden_size = 4
embed_size = 4
transformer_encoder = TransformerContextEncoder(64,4)


num_layers = 10
base_dist = ConditionalDiagonalNormal(shape=[2], 
#                                      context_encoder=MLP([1], [4], hidden_sizes=[32,32]))
                                        context_encoder=nn.Sequential(nn.Linear(4,32), nn.ReLU(), nn.Linear(32,4)))
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

flow = Flow(transform, base_dist, embedding_net=transformer_encoder)
optimizer = optim.Adam(flow.parameters(), lr=1e-3)


num_iter = 5000
loss_arr = []
for i in range(num_iter):
    indices = torch.randperm(feats.shape[0])[:2048].to(int)
    x = inputs[indices, :]
    ctx = feats[indices, :, :]

    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=ctx).mean()
    loss_arr.append(loss.detach())
    loss.backward()
    print(f"Iteration {i+1}/{num_iter} - Loss: {loss.detach().item()}")
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        flow.eval()
        xline = torch.linspace(-4.0, 4.0, 200)
        yline = torch.linspace(-4.0, 4.0, 200)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
        
        with torch.no_grad():
            ctx_test = feats[100, :, :]
            print(ctx_test)
            zgrid = flow.log_prob(xyinput, ctx_test.repeat(40000, 1, 1)).exp().reshape(200, 200)

        new_xline = (xline*pos_std[0])+pos_mean[0]
        new_yline = (yline*pos_std[1])+pos_mean[1]
        new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

        plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), levels = 100, cmap="viridis")
        x = (x*pos_std)+pos_mean

        m = targets[100] * pos_std + pos_mean
        m = m.numpy()
        plt.scatter(m[0], m[1], color='r', s=20)
        plt.title('iteration {}'.format(i + 1))
        plt.savefig(f"./figs/experiments/transformer/training_{i}.png")
        flow.train()
#%%
#save loss array
np.savetxt("figs/loss/transformer_loss.csv",loss_arr)

#%% 
#evaluate the log likelihood of the model
flow.eval()
idx = torch.arange(feats.shape[0])
idx_list = torch.split(idx,8192)
sum_log_probs = 0
for ix in tqdm(idx_list):
    x = inputs[ix, :]
    ctx = feats[ix, :, :]
    with torch.inference_mode():
        log_prob = flow.log_prob(inputs=x, context=ctx).detach().sum().item()
    sum_log_probs += log_prob

mean_log_probs = sum_log_probs/idx.shape[0]
print(f"Model Score: {mean_log_probs}")
flow.train()
print("stop")

#%%
# Make a gif
flow.eval()
seq_idx = 6
test_obs = obs_seqs[seq_idx]
test_pos = pos_seqs[seq_idx]
test_times = time_seqs[seq_idx]

xline = torch.linspace(-4.0, 4.0, 200)
yline = torch.linspace(-4.0, 4.0, 200)
xgrid, ygrid = torch.meshgrid(xline, yline)
xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
new_xline = (xline*pos_std[0])+pos_mean[0]
new_yline = (yline*pos_std[1])+pos_mean[1]
new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

ctx_test = torch.zeros(5, 3, dtype=torch.float32)

for i in range(min_seq_len, seq_lens[seq_idx]):    
    with torch.no_grad():
        ctx_test[:, 0:2] = test_obs[(i-min_seq_len):i, :]
        ctx_test[:, 2] = time_seqs[seq_idx][i-min_seq_len:i]
        zgrid = flow.log_prob(xyinput, ctx_test.repeat(40000, 1, 1)).exp().reshape(200, 200)

    plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), levels = 500, cmap="viridis")

    state_i = (test_pos[:i, :] * pos_std + pos_mean).numpy()
    obs_i = (test_obs[:i, :] * pos_std + pos_mean).numpy()
    plt.scatter(obs_i[:, 0], obs_i[:, 1], color='k', s=3)
    plt.plot(state_i[:, 0], state_i[:, 1], '--', color='r')
    plt.title('t = {}'.format(i + 1))
    plt.xlim((0, 100))
    plt.ylim((-10, 60))
    plt.savefig("./figs/experiments/transformer/tmp/tmp_{:03d}.png".format(i))
    # plt.show()

make_gif("./figs/experiments/transformer/tmp/", "./figs/experiments/transformer/seq6.gif", delete_frames=True)

#%%
# Make a gif
flow.eval()
seq_idx = 12
test_obs = obs_seqs[seq_idx]
test_pos = pos_seqs[seq_idx]
test_times = time_seqs[seq_idx]

xline = torch.linspace(-4.0, 4.0, 200)
yline = torch.linspace(-4.0, 4.0, 200)
xgrid, ygrid = torch.meshgrid(xline, yline)
xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
new_xline = (xline*pos_std[0])+pos_mean[0]
new_yline = (yline*pos_std[1])+pos_mean[1]
new_xgrid, new_ygrid = torch.meshgrid(new_xline, new_yline)

ctx_test = torch.zeros(5, 3, dtype=torch.float32)

for i in range(min_seq_len, seq_lens[seq_idx]):    
    with torch.no_grad():
        ctx_test[:, 0:2] = test_obs[(i-min_seq_len):i, :]
        ctx_test[:, 2] = time_seqs[seq_idx][i-min_seq_len:i]
        zgrid = flow.log_prob(xyinput, ctx_test.repeat(40000, 1, 1)).exp().reshape(200, 200)

    plt.contourf(new_xgrid.numpy(), new_ygrid.numpy(), zgrid.numpy(), levels = 500, cmap="viridis")

    state_i = (test_pos[:i, :] * pos_std + pos_mean).numpy()
    obs_i = (test_obs[:i, :] * pos_std + pos_mean).numpy()
    plt.scatter(obs_i[:, 0], obs_i[:, 1], color='k', s=3)
    plt.plot(state_i[:, 0], state_i[:, 1], '--', color='r')
    plt.title('t = {}'.format(i + 1))
    plt.xlim((0, 100))
    plt.ylim((-10, 60))
    plt.savefig("./figs/experiments/transformer/tmp/tmp_{:03d}.png".format(i))
    # plt.show()

make_gif("./figs/experiments/transformer/tmp/", "./figs/experiments/transformer/seq12.gif", delete_frames=True)
