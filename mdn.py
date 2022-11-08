# Adapted from https://github.com/sagelywizard/pytorch-mdn

"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from utils import read_dataset, read_bicycle_dataset
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)

def plot_ellipse(mu, sigma, alpha=1):
    print(sigma)
    lambda_, v = np.linalg.eig(sigma)
    lambda_ = np.sqrt(lambda_)
    ax = plt.gca()

    ell = Ellipse(xy=(mu[0], mu[1]),
                width=lambda_[0]*2, height=lambda_[1]*2,
                angle=np.rad2deg(np.arccos(v[0, 0])), edgecolor='r', alpha=alpha)
    ell.set_facecolor('None')
    return ell

def plot_mdn_prediction(ax, mu, sigma, pi):
    """
    mu: GXO
    sigma: GXO
    """
    ng = mu.shape[0]
    for i in range(ng):
        S = torch.diag(sigma[i, :])
        M = mu[i, :]
        print(S)
        print(M)

        if pi[i] == torch.max(pi):
            alpha = 1.0
        else:
            alpha = 0.5

        ell = plot_ellipse(M, S, alpha=alpha)
        ax.add_patch(ell)

    return ax

def plot_mdn_bicycle(mu, sigma, pi):
    fig = plt.figure()
    ax = plt.gca()
    nt = mu.shape[0]
    for i in range(nt):
        mi = mu[i, :, :]
        si = sigma[i, :, :]
        pii = pi[i, :]
        ax = plot_mdn_prediction(ax, mi, si, pii)

    return ax

data = read_bicycle_dataset("bicycle_dataset.h5")
train_feats = torch.unsqueeze(data[:, 2], dim=1)

feat_m = torch.mean(train_feats)
feat_s = torch.std(train_feats)
train_feats = (train_feats - feat_m)/feat_s

train_targets = data[:, 0:2]
target_m = torch.mean(train_targets, dim=0)
target_s = torch.std(train_targets, dim=0)
train_targets = (train_targets - target_m)/target_s

# initialize the model
model = nn.Sequential(
    nn.Linear(1, 8),
    nn.Tanh(),
   MDN(8, 2, 5)
)
optimizer = optim.Adam(model.parameters())

# train the model
nepochs = 5000
loss_history = torch.zeros(nepochs)
for epoch in range(nepochs):
    model.zero_grad()
    pi, sigma, mu = model(train_feats)
    loss = mdn_loss(pi, sigma, mu, train_targets)
    loss_history[epoch] = loss.item()
    loss.backward()
    optimizer.step()
    #if epoch % 10 == 99:
    print(f'{round(epoch)}', end='\n')

print('Done')

x_test = (torch.linspace(0, 15, 5).reshape(-1, 1) - feat_m)/feat_s
with torch.no_grad():
    pi, sigma, mu = model(x_test)

mu = mu*target_s + target_m
sigma = sigma*target_s

ax = plot_mdn_bicycle(mu, sigma, pi)

plt.gca()
plt.xlim(0, 90)
plt.ylim(0, 50)

plt.scatter(data[:, 0], data[:, 1], zorder=0, s=0.5)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid()
plt.show()


xs = torch.linspace(0, 90, steps=100)
ys = torch.linspace(0, 50, steps=100)
#x, y = torch.meshgrid(xs, ys, indexing='xy')

# for x in xs:
#     for y in ys:

