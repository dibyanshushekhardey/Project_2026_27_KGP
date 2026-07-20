import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions import Categorical
from tqdm import trange
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def median_heuristic(X):
  if X.dim() == 1:
    X = X.reshape(-1, 1)
  # x_i = X[:, None, :]
  # x_j = X[None, :, :]
  # dists_sq = ((x_i - x_j)**2).sum(dim=-1)
  dists_sq = torch.cdist(X,X, p=2) ** 2
  H_n = torch.median(dists_sq)
  h = torch.sqrt(H_n / 2.0)
  return h

def sample_gmm(n, means, sigmas, weights, device):
    means = torch.tensor(means,dtype=torch.float32, device=device)
    sigmas = torch.tensor(sigmas, dtype=torch.float32,device=device)
    weights = torch.tensor(weights, dtype=torch.float32,device=device)
    idx = torch.multinomial(weights, n, replacement=True)
    x = torch.randn(n, device=device) * sigmas[idx] + means[idx] # generate samples
    return x.unsqueeze(-1)