import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import torch
import h5py

def color_grid(num_x1, num_x2, x1_min, x1_max, x2_min, x2_max):
    x1 = np.linspace(x1_min, x1_max, num_x1)
    x2 = np.linspace(x2_min, x2_max, num_x2)
    X1,X2 = np.meshgrid(x1,x2)
    X = np.stack((X1,X2),axis=-1)
    X_flat = np.reshape(X,(-1,2))


    xA = [x1_min,x2_min]
    xB = [x1_max,x2_min]
    xC = [x1_max,x2_max]
    xD = [x1_min,x2_max]
    XS = np.stack((xA,xB,xC,xD)).T

    A = np.array([250, 166, 10])/255 #orange
    B = np.array([17, 95, 250])/255 #blue
    C = np.array([247, 12, 36])/255 # red
    D = np.array([7, 242, 133])/255 #green

    Z = np.stack((A,B,C,D))

    r_interp = interp2d(XS[0],XS[1],Z[:,0])
    r = np.array([r_interp(p[0],p[1]) for p in X_flat]).squeeze()

    g_interp = interp2d(XS[0],XS[1],Z[:,1])
    g = np.array([g_interp(p[0],p[1]) for p in X_flat]).squeeze()

    b_interp = interp2d(XS[0],XS[1],Z[:,2])
    b = np.array([b_interp(p[0],p[1]) for p in X_flat]).squeeze()

    color = np.stack((r,g,b),axis=-1)

    return X_flat, color

def color_dataset(x):
    x1_min = torch.min(x[:,0])
    x1_max = torch.max(x[:,0])
    x2_min = torch.min(x[:,1])
    x2_max = torch.max(x[:,1])

    xA = [x1_min,x2_min]
    xB = [x1_max,x2_min]
    xC = [x1_max,x2_max]
    xD = [x1_min,x2_max]
    XS = np.stack((xA,xB,xC,xD)).T

    A = np.array([250, 166, 10])/255 #orange
    B = np.array([17, 95, 250])/255 #blue
    C = np.array([247, 12, 36])/255 # red
    D = np.array([7, 242, 133])/255 #green

    Z = np.stack((A,B,C,D))

    r_interp = interp2d(XS[0],XS[1],Z[:,0])
    r = np.array([r_interp(p[0],p[1]) for p in x]).squeeze()

    g_interp = interp2d(XS[0],XS[1],Z[:,1])
    g = np.array([g_interp(p[0],p[1]) for p in x]).squeeze()

    b_interp = interp2d(XS[0],XS[1],Z[:,2])
    b = np.array([b_interp(p[0],p[1]) for p in x]).squeeze()

    color = np.stack((r,g,b),axis=-1)

    return color

def create_dataset(num_samples, fname):
    mu_1 = torch.Tensor([[0.5,3.2]])
    cov_1 = torch.Tensor([[0.4,-0.3],[-0.3,0.9]])
    mu_2 = torch.Tensor([[2.2,0.2]])
    cov_2 = torch.Tensor([[0.7,0.05],[0.05,0.1]])

    samples = []

    for _ in range(num_samples):
        switch = torch.rand((1,))
        switch = switch<0.65
        if switch == 0:
            samples.append(torch.distributions.multivariate_normal.MultivariateNormal(mu_1,cov_1).sample((1,)).squeeze())

        elif switch == 1:
            samples.append(torch.distributions.multivariate_normal.MultivariateNormal(mu_2,cov_2).sample((1,)).squeeze())

        else:
            raise ValueError
    
    samples = torch.stack(samples)
    with h5py.File(fname, 'w') as f:
        f.create_dataset('train', data=samples)

def read_dataset(fname):
    with h5py.File(fname, 'r') as f:
        dataset = np.array(f.get("train"))
    return torch.Tensor(dataset)

def read_bicycle_dataset(fname):
    data = h5py.File(fname, 'r')
    for group in data.keys() :
        print (group)
        for dset in data[group].keys():      
            arr = data[group][dset][:] # adding [:] returns a numpy array
    
    return torch.Tensor(arr).T

def reference(x):
    weights = [0.35, 0.65]
    mu_1 = torch.Tensor([[0.5,3.2]])
    cov_1 = torch.Tensor([[0.4,-0.3],[-0.3,0.9]])
    mu_2 = torch.Tensor([[2.2,0.2]])
    cov_2 = torch.Tensor([[0.7,0.05],[0.05,0.1]])
    dist1 = torch.distributions.multivariate_normal.MultivariateNormal(mu_1,cov_1)
    dist2 = torch.distributions.multivariate_normal.MultivariateNormal(mu_2,cov_2)
    probs1 = torch.exp(dist1.log_prob(x))
    probs2 = torch.exp(dist2.log_prob(x))
    return weights[0] * probs1 + weights[1] * probs2


