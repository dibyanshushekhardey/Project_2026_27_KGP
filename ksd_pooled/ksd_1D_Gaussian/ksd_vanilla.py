import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions import Categorical
from tqdm import trange
import pandas as pd

from functions import median_heuristic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#@title KSD Test no perturbation
class KSDTest:

  def __init__(self, means, sigmas, weights, device):
    self.device = device
    self.means = torch.as_tensor(means, dtype=torch.float32, device=device)
    self.sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
    self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

  def log_prob(self, x):
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    means = self.means.view(1, -1)
    sigmas = self.sigmas.view(1, -1)
    weights = self.weights.view(1, -1)
    log_components = (torch.log(weights)-0.5 * torch.log(2 * torch.pi * sigmas**2) -(x - means)**2/(2*sigmas**2))
    logp = torch.logsumexp(log_components, dim=1)
    return logp.reshape(batch_shape)

  def score_function(self, x):
    x = x.clone().detach().requires_grad_(True)
    logp = self.log_prob(x)
    score = torch.autograd.grad(logp.sum(), x,create_graph=True)[0]
    return score

  def rbf_kernel(self, x1, x2, scale):
    diff = x1 - x2
    return torch.exp(-(diff**2).sum(dim=-1) /(2*scale**2))
  def grad_x1_rbf(self, x1, x2, scale):
    k = self.rbf_kernel(x1, x2, scale)
    return ( -k.unsqueeze(-1)* (x1-x2) / scale**2)

  def grad_x2_rbf(self, x1, x2, scale):
    return -self.grad_x1_rbf( x1, x2, scale)

  def grad_x1_x2_rbf(self, x1, x2, scale):
    diff = x1-x2
    sq = (diff**2).sum(dim=-1)
    k = self.rbf_kernel(x1,x2,scale)
    d = diff.shape[-1]
    return (d/scale**2 -  sq/scale**4)*k

  def u_q(self, x1, x2, scale):
    score1 = self.score_function(x1)
    score2 = self.score_function(x2)
    k = self.rbf_kernel(x1,x2,scale)
    grad1 = self.grad_x1_rbf(x1,x2,scale)
    grad2 = self.grad_x2_rbf( x1,x2,scale)
    grad12 = self.grad_x1_x2_rbf(x1,x2,scale)
    return ((score1*score2).sum(dim=-1)*k + (score1*grad2).sum(dim=-1) + (grad1*score2).sum(dim=-1) + grad12)

  def compute_ksd(self, samples, scale):
    samples = samples.to(self.device)
    if samples.ndim == 1:
        samples = samples.unsqueeze(-1)
    n = samples.shape[0]
    x_i = samples[:,None,:]
    x_j = samples[None,:,:]
    U = self.u_q(x_i,x_j,scale)
    total = U.sum() - torch.diagonal(U).sum()
    ksd = total/(n*(n-1))
    return ksd,U

  def bootstrap_stat(self, U):
    n = U.shape[0]
    probs = torch.ones(n, device=self.device) / n
    samples = torch.multinomial(probs, num_samples=n, replacement=True)
    w = torch.bincount(samples, minlength=n).float()
    w_centered = (w / n) - (1.0 / n)
    S = w_centered @ U @ w_centered
    diag_correction = torch.sum(w_centered ** 2 * torch.diag(U))
    return S - diag_correction

  def compute_p_value_ksd(self, x, num_boot):
      #x = sample_gmm(n=n, means=means, sigmas=sigmas, weights=weights, device=self.device)
      scale = median_heuristic(x)
      scale = torch.clamp(scale, min=1e-3)
      ksd, U = self.compute_ksd(x, scale)
      boot_stats = torch.stack([self.bootstrap_stat(U) for _ in range(num_boot)])
      pvalue = (boot_stats >= ksd).float().mean().item()
      return pvalue