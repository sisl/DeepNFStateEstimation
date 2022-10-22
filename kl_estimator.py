import torch

def one_sample_kl_estimator(x1,dist,m):

    d = x1.shape[1]
    n = x1.shape[0]
    assert m>=n, "m<n does not yield a good approximation"

    k = 2   # as by the paper

    x2 = dist.sample((m,))

    x1a = x1.unsqueeze(0)
    x1b = x1.unsqueeze(1)
    x2a = x2.unsqueeze(0)


    diff1 = x1a - x1b
    diff2 = x2a - x1b

    distance1 = torch.linalg.norm(diff1,dim=-1)
    distance2= torch.linalg.norm(diff2,dim=-1)

    r_k = torch.topk(distance1,k,dim=1,largest=False).values[:,-1]
    s_k = torch.topk(distance2,k-1,dim=1,largest=False).values[:,-1]



    dkl = -torch.log(r_k/s_k).sum() * d / n + torch.log(torch.Tensor([m / (n - 1.)]))
    return dkl

def two_sample_kl_estimator(x1,x2):

    d = x1.shape[1]
    n = x1.shape[0]
    m = x2.shape[0]

    k = 2   # as by the paper

    x1a = x1.unsqueeze(0)
    x1b = x1.unsqueeze(1)
    x2a = x2.unsqueeze(0)


    diff1 = x1a - x1b
    diff2 = x2a - x1b

    distance1 = torch.linalg.norm(diff1,dim=-1)
    distance2= torch.linalg.norm(diff2,dim=-1)

    r_k = torch.topk(distance1,k,dim=1,largest=False).values[:,-1]
    s_k = torch.topk(distance2,k-1,dim=1,largest=False).values[:,-1]



    dkl = -torch.log(r_k/s_k).sum() * d / n + torch.log(torch.Tensor([m / (n - 1.)]))
    return dkl

if __name__=="__main__":
    dist1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((2,)),torch.eye(2))
    # dist1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.Tensor([2.3,3.7]),torch.Tensor([[5.9,-0.9],[-0.9,10.9]]))
    dist2 = torch.distributions.multivariate_normal.MultivariateNormal(torch.Tensor([0.5,-0.7]),torch.Tensor([[0.9,0.5],[0.5,1.9]]))
    
    x = dist2.sample((1000,))
    print(one_sample_kl_estimator(x,dist1,1500))









