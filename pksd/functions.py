#@title Generating the samples function
import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions import Categorical
from tqdm import trange
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample_gmm(n, means, sigmas, weights, device):
    means = torch.tensor(means,dtype=torch.float32, device=device)
    sigmas = torch.tensor(sigmas, dtype=torch.float32,device=device)
    weights = torch.tensor(weights, dtype=torch.float32,device=device)
    idx = torch.multinomial(weights, n, replacement=True)
    #return torch.randn(n, device=device) * sigmas[idx] + means[idx]

    x = torch.randn(n, device=device) * sigmas[idx] + means[idx] # generate samples

    return x.unsqueeze(-1)

#@title Median heuristic for scaling in RBF Kernel https://arxiv.org/pdf/1707.07269

def median_heuristic(X):
  if X.dim() == 1:
    X = X.reshape(-1, 1)
  dists_sq = torch.cdist(X, X, p=2) ** 2
  # remove diagonal zeros if desired
  mask = ~torch.eye(X.shape[0], dtype=torch.bool, device=X.device)
  vals = dists_sq[mask]
  H_n = torch.median(vals)
  h = torch.sqrt(H_n / 2.0)
  return h