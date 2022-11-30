import torch
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import math

n_samples = 1000
confidence_level = 0.95

# Set values for the mixture
pi = [0.6, 0.3, 0.1]
means = [0., 10., -8.2]
sigmas = [2., 1., 0.8]

gm = MixtureSameFamily(mixture_distribution=Categorical(probs=torch.tensor(pi)),
                       component_distribution=Normal(loc=torch.tensor(means), scale=torch.tensor(sigmas)))


samples = gm.sample(sample_shape=torch.Size([1000]))

log_probs = gm.log_prob(samples)

idx = torch.argsort(log_probs,descending=True)

cutoff = int(n_samples*confidence_level)

log_prob_crit = log_probs[idx[cutoff]]

print("stop")