import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions import Categorical
from tqdm import trange
import pandas as pd

from ksd_pooled.ksd_1D_Gaussian.functions import median_heuristic
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#@title Perturbed Class KSD

class PooledKSD:
  def __init__(self, means, sigmas, weights, device):
    self.device = device
    self.means = torch.as_tensor(means, dtype=torch.float32, device=device)
    self.sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
    self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

  def log_prob(self, x, sigma_noise=0.0):
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    means = self.means.view(1, -1)
    sigmas = self.sigmas.view(1, -1)
    weights = self.weights.view(1, -1)
    #effective variance after perturbation
    var = sigmas**2 + sigma_noise**2
    log_components = (torch.log(weights)-0.5 * torch.log(2 * torch.pi * var) -(x - means)**2/(2*var))
    logp = torch.logsumexp(log_components, dim=1)
    return logp.reshape(batch_shape)

  def score_function(self, x, sigma_noise=0.0):
    x = x.clone().detach().requires_grad_(True)
    logp = self.log_prob(x, sigma_noise)
    score = torch.autograd.grad(logp.sum(),x,create_graph=True)[0]
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

  def u_q(self, x1, x2,scale,sigma_l):
    score1 = self.score_function(x1, sigma_l)
    score2 = self.score_function(x2, sigma_l)
    k = self.rbf_kernel(x1,x2,scale)
    grad1 = self.grad_x1_rbf(x1,x2,scale)
    grad2 = self.grad_x2_rbf( x1,x2,scale)
    grad12 = self.grad_x1_x2_rbf(x1,x2,scale)
    return ((score1*score2).sum(dim=-1)*k + (score1*grad2).sum(dim=-1) + (grad1*score2).sum(dim=-1) + grad12)

  def bootstrap_stat(self, U):
    n = U.shape[0]
    probs = torch.ones(n, device=self.device) / n
    samples = torch.multinomial(probs, num_samples=n, replacement=True)
    w = torch.bincount(samples, minlength=n).float()
    w_centered = (w / n) - (1.0 / n)
    S = w_centered @ U @ w_centered
    diag_correction = torch.sum(w_centered ** 2 * torch.diag(U))
    return S - diag_correction
  
  def reject(self, boot_stats, ksd,alpha):
    critical_val = torch.quantile(boot_stats, 1 - alpha)
    reject = (ksd > critical_val).int()
    return reject
  
  def pooled_u_stat(self, sigma_list, x_ksd):
    U = 0
    for sigma_l in sigma_list:
      epsilon = torch.randn_like(x_ksd)
      x_i_tilde = x_ksd + sigma_l * epsilon
      x1 = x_i_tilde[:, None, :]
      x2 = x_i_tilde[None, :, :]
      scale = median_heuristic(x_ksd) # Currently I am passing unperturbed sample, should it be perturbed sample?
      scale = torch.clamp(scale, min=1e-3)
      u_q_sigma_L = self.u_q(x1,x2,scale,sigma_l)
      U += u_q_sigma_L
    return U
  
  def pooled_ksd(self, U, n):
    total = U.sum() - torch.diagonal(U).sum()
    pooled_ksd = total/(n*(n-1))
    return total, pooled_ksd