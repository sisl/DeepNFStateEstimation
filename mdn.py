# Adapted from https://github.com/sagelywizard/pytorch-mdn

"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from utils import read_dataset, read_bicycle_dataset
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
import h5py



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

def plot_density_discrete(model, ts, feat_m, feat_s, target_m, target_s, n=500, levels=10):
    # grid the space
    model.eval()
    x = np.linspace(0, 90, n)
    y = np.linspace(0, 50, n)
    xx, yy = np.meshgrid(x, y)
    eval_targets = torch.Tensor(np.vstack([xx.flatten(), yy.flatten()]).T).squeeze()
    norm_eval_targets = (eval_targets - target_m)/target_s

    t_eval = (torch.Tensor(ts).unsqueeze(dim=1) - feat_m)/feat_s
    with torch.no_grad():
        pi, sigma, mu = model(t_eval)

    mix = Categorical(pi.squeeze())
    comp = torch.distributions.Independent(Normal(mu.squeeze(), sigma.squeeze()), 1)
    gmm = MixtureSameFamily(mix, comp)
    log_prob_1s = mc_level_set(gmm, 0.68, 5000000)
    log_prob_2s = mc_level_set(gmm, 0.95, 5000000)
    log_prob_3s = mc_level_set(gmm, 0.997, 5000000)
    levels = np.exp(np.array([log_prob_3s, log_prob_2s, log_prob_1s]))

    # total_density = np.zeros((n, n, len(ts)))
    total_density = np.zeros((n, n))
    for (i, t) in enumerate(ts):
        eval_feats = t * torch.ones(n*n, 1)#torch.Tensor([[10.0]])
        norm_eval_feats =  (eval_feats - feat_m)/feat_s

        pi_eval, sigma_eval, mu_eval = model(norm_eval_feats)

        probs = torch.sum(pi_eval * gaussian_probability(sigma_eval, mu_eval, norm_eval_targets), dim=1)
        nll = -torch.log(probs)
        nll = nll.reshape((n, n))
        probs = probs.reshape((n, n))
        # total_density[:, :, i] = probs.detach().numpy()
        total_density +=  probs.detach().numpy()
    print(np.max(total_density))
    print(np.min(total_density))
    plt.rcParams['text.usetex'] = True
    fig = plt.figure()
    cn = plt.contour(xx, yy, total_density, levels=levels, vmin=levels[0], vmax=levels[2], norm=matplotlib.colors.LogNorm())#
    return cn
    #plt.show()

def extract_contours(cn):
    contours = []
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)
    with h5py.File("flow_level_sets_mdn.h5", 'w') as f:
        f.create_dataset('x68', data=contours[0][0][:,0])
        f.create_dataset('y68', data=contours[0][0][:,1])
        f.create_dataset('x95', data=contours[1][0][:,0])
        f.create_dataset('y95', data=contours[1][0][:,1])
        f.create_dataset('x995', data=contours[2][0][:,0])
        f.create_dataset('y995', data=contours[2][0][:,1])

def mc_level_set(gm, confidence_level, n_samples):
    samples = gm.sample(sample_shape=torch.Size([n_samples]))
    log_probs = gm.log_prob(samples)
    idx = torch.argsort(log_probs,descending=True)
    cutoff = int(n_samples*confidence_level)
    log_prob_crit = log_probs[idx[cutoff]]
    return log_prob_crit


def main(train=False, savepath = "./logs/mdn/mdn.pt"):
    fname = "bicycle_dataset_continuous.h5"
    data = h5py.File(fname, 'r')

    train_feats = torch.unsqueeze(torch.Tensor(data["time"][:]),  dim=1)
    print(train_feats.shape)

    feat_m = torch.mean(train_feats)
    feat_s = torch.std(train_feats)
    train_feats = (train_feats - feat_m)/feat_s

    train_targets = torch.Tensor(data["position"][:].T)
    print(train_targets.shape)
    target_m = torch.mean(train_targets, dim=0)
    target_s = torch.std(train_targets, dim=0)
    train_targets = (train_targets - target_m)/target_s

    # initialize the model
    model = nn.Sequential(
        nn.Linear(1, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.Tanh(),
    MDN(8, 2, 10)
    )

    if train:
        optimizer = optim.Adam(model.parameters(), lr=5e-4)

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
            #print(f'{round(epoch)}', end='\n')

        print('Done')

        # x_test = (torch.linspace(0, 15, 5).reshape(-1, 1) - feat_m)/feat_s
        x_test = (torch.arange(0, 16, 2).reshape(-1, 1) - feat_m)/feat_s
        with torch.no_grad():
            pi, sigma, mu = model(x_test)

        mu = mu*target_s + target_m
        sigma = sigma*target_s

        ax = plot_mdn_bicycle(mu, sigma, pi)
        plt.gca()
        plt.xlim(0, 90)
        plt.ylim(0, 70)

        #raw_position = data["position"][:].T
        discrete_data = h5py.File("bicycle_dataset_discrete.h5", 'r')
        plt.scatter(discrete_data["position"][0, :], discrete_data["position"][1, :], zorder=0, s=0.5)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid()
        #plt.show()

        plt.figure()
        plt.plot(loss_history)
        #plt.show()

        torch.save(model.state_dict(), savepath)


    model.load_state_dict(torch.load(savepath))

    # grid the space
    n = 500
    x = np.linspace(0, 90, n)
    y = np.linspace(0, 50, n)
    xx, yy = np.meshgrid(x, y)

    eval_feats = 5.0 * torch.ones(n*n, 1)#torch.Tensor([[10.0]])
    norm_eval_feats =  (eval_feats - feat_m)/feat_s

    eval_targets = torch.Tensor(np.vstack([xx.flatten(), yy.flatten()]).T).squeeze()
    norm_eval_targets = (eval_targets - target_m)/target_s

    pi_eval, sigma_eval, mu_eval = model(norm_eval_feats)

    probs = torch.sum(pi_eval * gaussian_probability(sigma_eval, mu_eval, norm_eval_targets), dim=1)
    nll = -torch.log(probs)
    print(probs.shape)
    nll = nll.reshape((n, n))
    probs = probs.reshape((n, n))

    plt.rcParams['text.usetex'] = True
    plt.figure()
    plt.contourf(xx, yy, probs.detach().numpy())
    plt.show()
    
    ts = [13]
    fig = plot_density_discrete(model, ts, feat_m, feat_s, target_m, target_s, n=500)
    extract_contours(fig)    
    plt.gcf()
    #plt.show()

    tfinal = 13.0
    #Try plotting the mean too
    t_mean = (torch.linspace(0.0, tfinal, 100).unsqueeze(dim=1) - feat_m)/feat_s

    with torch.no_grad():
        pi_t, sigma_t, mu_t = model(t_mean)

    mu_temp = [pi_t[i] * mu_t[i, :, :].T for i in range(pi_t.shape[0])]
    print(len(mu_temp))
    mu_total = torch.vstack([torch.sum(m, dim=1) for m in mu_temp])
    mu_total = mu_total*target_s + target_m

    #plt.figure()
    plt.plot(mu_total[:, 0], mu_total[:, 1], color='k', linewidth=2)
    plt.scatter(data["position"][0, :], data["position"][1, :], s=0.1, color="gray")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0, 90])
    plt.ylim([0, 60])
    plt.show()

    print(data["position"].shape)
    #plt.scatter()



if __name__ == "__main__":
    main(train=False, savepath="./logs/mdn/mdn.pt")