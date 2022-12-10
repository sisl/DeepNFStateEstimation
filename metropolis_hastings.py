import torch

def metropolis_hastings(w,model,num_samples=100,num_start=100):
    

    x = torch.Tensor([0,0])
    xs = []

    for i in range(num_start):
        proposal = torch.distributions.multivariate_normal.MultivariateNormal(x.squeeze(),0.1 * torch.eye(2))
        xp = proposal.sample((1,))
        alpha = min(1,w(xp.squeeze(),model)/w(x.squeeze(),model))
        coin = torch.rand(1)
        if coin <= alpha:
            x = xp
            # xs.append(xp.squeeze())
    
    while len(xs) < num_samples:
        proposal = torch.distributions.multivariate_normal.MultivariateNormal(x.squeeze(),0.1 * torch.eye(2))
        xp = proposal.sample((1,))
        alpha = min(1,w(xp,model)/w(x,model))
        coin = torch.rand(1)
        if coin <= alpha:
            x = xp
            xs.append(xp.squeeze())
        
    
    return xs
