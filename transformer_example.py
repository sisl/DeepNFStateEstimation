import torch
import torch.nn as nn
import copy
import random
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt

from transformer_utils import *
  
with h5py.File("bicycle_dataset_continuous.h5", 'r') as f:
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

for (i, o) in enumerate(obs_seqs):
    #feats += [o[0:j, :] for j in range(min_seq_len, o.shape[0])]
    # feats += [torch.cat((o[0:j, :], time_seqs[i][0:j].unsqueeze(0).T), dim=1) for j in range(min_seq_len, o.shape[0])]
    feats += [torch.cat((o, time_seqs[i].unsqueeze(0).T), dim=1)]
    # targets += [pos_seqs[i][k, :] for k in range(min_seq_len, o.shape[0])]

feat_lens = torch.Tensor([f.shape[0] for f in feats])


# pad/pack features
pad_val = -1.0
feats_padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=batch_first, padding_value=pad_val).to(torch.float32)
# feats_packed = torch.nn.utils.rnn.pack_padded_sequence(feats_padded, feat_lens, batch_first=batch_first, enforce_sorted=False)

# inputs = torch.vstack(targets).to(torch.float32)

#########################

train_data = generate_random_data(9000)
val_data = generate_random_data(3000)

train_dataloader = batchify_data(train_data)
val_dataloader = batchify_data(val_data)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10)

plt.plot(train_loss_list, label = "Train loss")
plt.plot(validation_loss_list, label = "Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()


# Here we test some examples to observe how the model predicts
examples = [
    torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
]

for idx, example in enumerate(examples):
    result = predict(model, example)
    print(f"Example {idx}")
    print(f"Input: {example.view(-1).tolist()[1:-1]}")
    print(f"Continuation: {result[1:-1]}")
    print()