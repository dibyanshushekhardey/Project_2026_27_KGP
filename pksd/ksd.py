#@title KSD Test class

import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions import Categorical
from tqdm import trange
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from pksd.functions import median_heuristic
from pksd.langevin import RandomWalkMH, prepare_proposal_input_all
from pksd.find_modes import find_modes, pairwise_directions
class KSDTest:

  def __init__(self, means, sigmas, weights=None,
                device="cuda" if torch.cuda.is_available() else "cpu"):
    self.device = device
    self.means = torch.as_tensor(means, dtype=torch.float32, device=device)
    self.sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
    if weights is None:
        weights = torch.ones(len(means), device=device)
        weights /= weights.sum()
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
    U = self.u_q(
        x_i,
        x_j,
        scale
    )
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

  def h1_var(self, X, return_scaled_ksd=False):
    scale = median_heuristic(X)
    scale = torch.clamp(scale, min=1e-6)
    _, u_mat = self.compute_ksd(X, scale)

    n = X.shape[0]
    witness = torch.sum(u_mat, dim=1)

    term1 = 4.0 * torch.sum(witness ** 2) / (n ** 3)
    term2 = 4.0 * (torch.sum(u_mat) ** 2) / (n ** 4)
    var = term1 - term2 + 1e-12
    if not return_scaled_ksd:
        return var
    else:
        ksd = (u_mat.sum() - torch.diagonal(u_mat).sum()) / (n * (n - 1))
        ksd_scaled = ksd / torch.sqrt(var)

        return var, ksd_scaled

  def compute_p_value_ksd(self, x, num_boot):
    #x = sample_gmm(n=n, means=means, sigmas=sigmas, weights=weights, device=self.device)
    scale = median_heuristic(x)
    scale = torch.clamp(scale, min=1e-3)
    ksd, U = ksd_test.compute_ksd(x, scale)
    boot_stats = torch.stack([self.bootstrap_stat(U) for _ in range(num_boot)])
    pvalue = (boot_stats >= ksd).float().mean().item()
    return pvalue

  # def compute_ksd(self, samples, scale):
  #   samples = samples.to(self.device)
  #   if samples.ndim == 1:
  #       samples = samples.unsqueeze(-1)
  #   n = samples.shape[0]
  #   x_i = samples[:, None, :]
  #   x_j = samples[None, :, :]
  #   U = self.u_q(
  #       x_i,
  #       x_j,
  #       scale,
  #   )
  #   # Zero the diagonal (TensorFlow implementation)
  #   U = U.clone()
  #   U.fill_diagonal_(0.0)
  #   ksd = U.sum() / (n * (n - 1))
  #   return ksd, U

  #@title PKSD Tests class
class PKSD(KSDTest):

    def __init__(self,means,sigmas,weights,jump_ls,T,num_starts=30,device=device):
        super().__init__(means, sigmas, weights, device)
        self.jump_ls = jump_ls
        self.T = T
        self.num_starts = num_starts
        self._build_proposal()

    def _build_proposal(self):
      start_pts = torch.randn(self.num_starts, 1, device=self.device) * 6
      mode_list, inv_hess_list = find_modes(start_pts=start_pts,log_prob_fn=self.log_prob,threshold=0.1,  max_iterations=1000)
      self.proposal_dict = prepare_proposal_input_all( mode_list,inv_hess_list,)
      if len(mode_list) == 1:
          self.ind_pair_list = [(0, 0)]
      else:
          _, self.ind_pair_list = pairwise_directions(  mode_list, return_index=True,)
      # debugging printing mode_list and inv_hessian list
      # print("Number of modes:", len(mode_list))
      # for i, mode in enumerate(mode_list):
      #   print(f"Mode {i}: {mode.cpu().numpy()}")
      # print(self.ind_pair_list)
      # for i, H in enumerate(inv_hess_list):
      #   print(i, H)

    def perturb_samples(self, x, jump):
      sampler = RandomWalkMH(self.log_prob)
      sampler.run(steps=self.T,x_init=x, std=torch.tensor([jump], device=self.device), ind_pair_list=self.ind_pair_list,**self.proposal_dict,)

      ### Changes
      #print("Acceptance rate:", sampler.if_accept.float().mean().item()) #. checking acceptance rate
      # print(
      #   f"jump={jump:.2f}, "
      #   f"accept={sampler.if_accept.float().mean().item():.4f}"
      # )
      movement = torch.norm(sampler.x[0, -1] - sampler.x[0, 0],dim=1,)
      # print(
      #     f"mean movement={movement.mean().item():.4f}, "
      #     f"max movement={movement.max().item():.4f}"
      # )
      return sampler.x[0, -1]

    def find_best_jump(self, x_train):
      best_jump = None
      best_stat = -float("inf")
      for jump in self.jump_ls:
          x_t = self.perturb_samples(x_train, jump.item())
          _, scaled = self.h1_var(X=x_t, return_scaled_ksd=True,)
          if scaled > best_stat:
            best_stat = scaled
            best_jump = jump.item()

      return best_jump

    def compute_p_value_pksd(self, x_train, x_test, num_boot):
      best_jump = self.find_best_jump(x_train)
      x_t = self.perturb_samples(x_test, best_jump)
      #print("Mean movement:", (x_t - x_test).abs().mean().item()) # checking mean movement
      #print("Max movement :", (x_t - x_test).abs().max().item())  # checking max movement
      scale = median_heuristic(x_t)
      scale = torch.clamp(scale, min=1e-3)
      ksd, U = self.compute_ksd(x_t, scale)
      boot_stats = torch.stack([self.bootstrap_stat(U) for _ in range(num_boot)])
      pvalue = (boot_stats >= ksd).float().mean().item()
      #best_jump = self.find_best_jump(x_train)
      #print("Best jump:", best_jump) # printing the best jump
      return pvalue
    
#@title OSPKSD Class

class OSPKSD(PKSD):

    def __init__(self, means,sigmas, weights, jump_ls,T,num_starts=30, device="cuda" if torch.cuda.is_available() else "cpu",):
        super().__init__(means=means,sigmas=sigmas, weights=weights, jump_ls=jump_ls, T=T,device=device,)

    # ospKSD statistic
    def compute_ospksd(self, x0, xt):
      X = torch.cat([x0, xt], dim=0)
      scale = median_heuristic(X)
      stat0, U0 = self.compute_ksd(x0, scale)
      stat1, U1 = self.compute_ksd(xt, scale)
      stat = stat0 + stat1
      U = U0 + U1
      return stat, U

    # scaled ospKSD used for selecting jump
    def h1_var(self, x0, xt, return_scaled_ksd=False):
      X = torch.cat([x0, xt], dim=0)
      scale = median_heuristic(X)
      stat0, U0 = self.compute_ksd(x0, scale)
      stat1, U1 = self.compute_ksd(xt, scale)
      U = U0 + U1
      n = x0.shape[0]
      witness = torch.sum(U, dim=1)
      term1 = 4.0 * torch.sum(witness ** 2) / (n ** 3)
      term2 = 4.0 * (torch.sum(U) ** 2) / (n ** 4)
      var = term1 - term2 + 1e-12
      if not return_scaled_ksd:
          return var
      stat = stat0 + stat1
      #stat = (U.sum() - torch.diagonal(U).sum()) / (n * (n - 1))
      stat_scaled = stat / torch.sqrt(var)
      return var, stat_scaled

    # choose jump
    def find_best_jump(self, x_train):
      best_jump = None
      best_value = -float("inf")
      for jump in self.jump_ls:
          xt = self.perturb_samples(x_train,jump.item(),)
          _, value = self.h1_var(x_train,xt,return_scaled_ksd=True,)
          if value > best_value:
              best_value = value
              best_jump = jump.item()
      return best_jump

    def compute_p_value_ospksd(self, x_train, x_test, num_boot):
      best_jump = self.find_best_jump(x_train)
      xt = self.perturb_samples(x_test, best_jump)
      ospksd, U = self.compute_ospksd(x_test, xt)
      boot_stats = torch.stack([self.bootstrap_stat(U) for _ in range(num_boot)])
      pvalue = (boot_stats >= ospksd).float().mean().item()
      return pvalue
    
#@title SPKSD Class
class SPKSD(PKSD):
  def __init__(self,means,sigmas,weights,jump_ls,T,num_starts=30,device="cuda" if torch.cuda.is_available() else "cpu",):
    super().__init__(means=means,sigmas=sigmas, weights=weights, jump_ls=jump_ls, T=T,device=device,)

  def compute_spksd(self, X):
    total_stat = 0.0
    total_U = None
    for i in range(X.shape[0]):
      scale = median_heuristic(X[i])
      stat, U = self.compute_ksd(X[i],scale,)
      total_stat += stat
      if total_U is None:
          total_U = U
      else:
          total_U += U

    return total_stat, total_U

  def perturb_all(self, x):
    perturbed = []
    for jump in self.jump_ls:
      xt = self.perturb_samples( x, jump.item(),)
      perturbed.append(xt)

    return torch.stack(perturbed,dim=0,)

  def compute_p_value_spksd(self, x, num_boot):
    xt = self.perturb_all(x)
    stat, U = self.compute_spksd(xt)
    boot_stats = torch.stack([self.bootstrap_stat(U)for _ in range(num_boot) ])
    pvalue = (boot_stats >= stat).float().mean().item()
    return pvalue