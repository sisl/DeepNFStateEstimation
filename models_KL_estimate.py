import h5py
from kl_estimator import two_sample_kl_estimator,kl_two_sample_alternative
from utils import read_bicycle_dataset
import numpy as np
import torch
import matplotlib.pyplot as plt

x_complete_all,t = read_bicycle_dataset("bicycle_dataset_continuous.h5")
x_conditioned = x_complete_all[t==13,:][:1000]


with h5py.File("samples_ukf.h5", 'r') as f:
    ukf_samples = torch.Tensor(np.array(f["samples"]))

dkl_ukf = two_sample_kl_estimator(x_conditioned,ukf_samples)
print("DKL_UKF: ",dkl_ukf.item())

with h5py.File("samples_mdn.h5", 'r') as f:
    mdn_samples = torch.Tensor(np.array(f["samples"]))

dkl_mdn = two_sample_kl_estimator(x_conditioned,mdn_samples)
print("DKL_MDN: ",dkl_mdn.item())

with h5py.File("samples_vae.h5", 'r') as f:
    vae_samples = torch.Tensor(np.array(f["samples"]))

dkl_vae = two_sample_kl_estimator(x_conditioned,vae_samples)
print("DKL_VAE: ",dkl_vae.item())

with h5py.File("samples_nflow.h5", 'r') as f:
    nflow_samples = torch.Tensor(np.array(f["samples"]))

dkl_nflow = two_sample_kl_estimator(x_conditioned,nflow_samples)
print("DKL_nflow: ",dkl_nflow.item())

