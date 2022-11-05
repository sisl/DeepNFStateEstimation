import gmm
from utils import read_dataset, read_bicycle_dataset
import torch
import matplotlib.pyplot as plt

# x = read_dataset("dataset2.h5")
x = read_bicycle_dataset("bicycle_dataset.h5")

model = gmm.GaussianMixture(80,3,"full")

# model.fit(x)
model.fit(x)

weights = torch.distributions.categorical.Categorical(model.pi.squeeze())
mu = model.mu.squeeze()
sigma = model.var.squeeze()
normal = torch.distributions.multivariate_normal.MultivariateNormal(mu,sigma)
mixture_model = torch.distributions.mixture_same_family.MixtureSameFamily(weights,normal)

x1 = torch.linspace(-5,100,100)
x2 = torch.linspace(-5,60,100)
X = torch.meshgrid((x1,x2))
X = torch.stack(X,dim=-1)
X_flat = torch.reshape(X,(-1,2))
X_flat = torch.cat((X_flat,3.0*torch.ones((X_flat.shape[0],1))),dim=1)
probs = torch.exp(mixture_model.log_prob(X_flat))
# probs = mixture_model.log_prob(X_flat)
probs = probs.reshape(X[:,:,-1].shape)

plt.rcParams['text.usetex'] = True
plt.contourf(X[:,:,0],X[:,:,1],probs.detach().numpy(),levels=50)
# plt.scatter(x[:,0],x[:,1],s=1,color=(0,0,0))
plt.xlim(-5,100)
plt.ylim(-5,60)
# plt.savefig(f"./figs/experiments/{base_name}/density_{i}.png",dpi=600)
plt.show()
plt.clf()

print("stop")
