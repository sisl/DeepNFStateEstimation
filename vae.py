import torch
import torch.nn as nn
from kl_estimator import one_sample_kl_estimator,two_sample_kl_estimator
from utils import read_dataset, read_bicycle_dataset
import os
from plot import plot_space_mapping,plot_dataset_mapping,plot_density

EXPERIMENT_NAME = "vae_1"

for fp in [f"./figs/experiments/{EXPERIMENT_NAME}", f"./figs/experiments/{EXPERIMENT_NAME}/train",f"./logs/{EXPERIMENT_NAME}"]:
    isExist = os.path.exists(fp)
    if not isExist:
        os.makedirs(fp)

class VAE(nn.Module):
    def __init__(self,input_dim) -> None:
        super().__init__()
        self.enc_layer1 = nn.Linear(input_dim,64)
        self.enc_layer2a = nn.Linear(64,64)
        self.enc_layer2b = nn.Linear(64,64)
        self.enc_layer2c = nn.Linear(64,64)
        self.enc_layer2d = nn.Linear(64,64)
        self.enc_layer2e = nn.Linear(64,64)
        self.enc_layer2f = nn.Linear(64,64)
        self.enc_layer3 = nn.Linear(64,input_dim)
        self.dec_layer1 = nn.Linear(input_dim,64)
        self.dec_layer2a = nn.Linear(64,64)
        self.dec_layer2b = nn.Linear(64,64)
        self.dec_layer2c = nn.Linear(64,64)
        self.dec_layer2d = nn.Linear(64,64)
        self.dec_layer2e = nn.Linear(64,64)
        self.dec_layer2f = nn.Linear(64,64)
        self.dec_layer3 = nn.Linear(64,input_dim)
          
    def encode(self,x):
        z = self.enc_layer1(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2a(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2b(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2c(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2d(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2e(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer2f(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        z = self.enc_layer3(z)
        
        return z

    def decode(self,z):
        x = self.enc_layer1(z)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2a(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2b(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2c(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2d(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2e(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer2f(x)
        # z = torch.nn.functional.leaky_relu(z)
        z = torch.tanh(z)
        x = self.enc_layer3(x)
    
        return x
    
    def forward(self,x):
        return self.encode(x)

input_dim = 2
epochs = 1000
model = VAE(input_dim)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0009)

x,t = read_bicycle_dataset("bicycle_dataset_discrete.h5")
x = x[t==14,:]
x = (x-torch.mean(x,dim=0))/torch.std(x,dim=0)

for i in range(epochs):
    optimizer.zero_grad()
    z_predict = model.encode(x)
    x_predict = model.decode(z_predict)
    latent_target_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((input_dim,)),torch.eye(input_dim))
    latent_loss = one_sample_kl_estimator(z_predict,latent_target_dist,5000)
    reconstruction_loss = torch.linalg.norm(x-x_predict,dim=1).mean()
    # reconstruction_loss = torch.nn.functional.mse_loss(x_predict,x)
    # reconstruction_loss = two_sample_kl_estimator(x,x_predict)
    loss = 5 * latent_loss + reconstruction_loss
    print(f"Epoch {i}/{epochs} - Loss: {loss.detach().item()}")
    loss.backward()
    optimizer.step()

plot_density(model,x,6767,EXPERIMENT_NAME)
plot_space_mapping(model,x,EXPERIMENT_NAME)
plot_dataset_mapping(model,x,EXPERIMENT_NAME,comparison=True)

print("stop")
