import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions import Categorical
from tqdm import trange
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from ksd_pooled.ksd_1D_Gaussian.ksd_vanilla import KSDTest
from ksd_pooled.ksd_1D_Gaussian.ksd_pooled import PooledKSD
from ksd_pooled.ksd_1D_Gaussian.functions import median_heuristic, sample_gmm

#@title Perturbed Power Tests with sigmas between 1 and 10 in geometric progression
sigma = 1
n = 1000
num_boot = 500
sigma_max = 10
sigma_min = 1
delta_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #
sigma_list = torch.logspace(start = torch.log10(torch.tensor(sigma_min)), end=torch.log10(torch.tensor(sigma_max)), steps=10)
n_trials = 100
rejection_rates_pooled_ksd = []
rejection_rates_ksd = []
alpha = 0.05
for delta in delta_list:
  # Vanilla KSD
  p_values_ksd = []
  # Target distribution q
  ksd_test = KSDTest(means=[0, delta],sigmas=[sigma, sigma], weights=[0.5, 0.5],device=device,)
  for trial in range(n_trials):
      # Sample distribution p (ratio_s = 1.0)
      x_ksd = sample_gmm(n=n,means=[0, delta],sigmas=[sigma, sigma], weights=[1.0, 0.0],device=device,) # all samples from left mode
      p_ksd = ksd_test.compute_p_value_ksd(x_ksd,num_boot,)
      p_values_ksd.append(p_ksd)

  rejection_rates_ksd.append(np.mean(np.array(p_values_ksd) < alpha))


  #Pooled KSD
  rejections = []
  pksd = PooledKSD(means=[0, delta],sigmas=[sigma, sigma], weights=[0.5, 0.5],device=device,)  # Target distribution p
  for trial in range(n_trials):
    # Sample distribution p (ratio_s = 1.0)
    x_ksd = sample_gmm(n=n,means=[0, delta],sigmas=[sigma, sigma], weights=[1, 0],device=device,) #all samples from left mode
    U = pksd.pooled_ksd(sigma_list, x_ksd)
    total, pooled_ksd = pksd.pooled_ksd(U, n)
    # U = 0
    # for sigma_l in sigma_list:
    #   epsilon = torch.randn_like(x_ksd)
    #   x_i_tilde = x_ksd + sigma_l * epsilon
    #   x1 = x_i_tilde[:, None, :]
    #   x2 = x_i_tilde[None, :, :]
    #   scale = median_heuristic(x_ksd) # Currently I am passing unperturbed sample, should it be perturbed sample?
    #   scale = torch.clamp(scale, min=1e-3)
    #   u_q_sigma_L = pksd.u_q(x1,x2,scale,sigma_l)
    #   U += u_q_sigma_L
    # total = U.sum() - torch.diagonal(U).sum()
    # ksd = total/(n*(n-1))
    boot_stats = torch.stack([pksd.bootstrap_stat(U) for _ in range(num_boot)])
    count = torch.count_nonzero(boot_stats >= pooled_ksd)
    # p_value = (count + 1).float() / (boot_stats.numel() + 1)
    critical_val = torch.quantile(boot_stats, 1 - alpha)
    reject = (pooled_ksd > critical_val).int()
    rejections.append(reject.item())

  rejection_rates_pooled_ksd.append(np.mean(rejections))

plt.figure(figsize=(6,4))
plt.plot(delta_list,rejection_rates_ksd,marker='x',linewidth=2,label='Vanilla KSD')
plt.plot(delta_list,rejection_rates_pooled_ksd,marker='o',linewidth=2,label='Pooled KSD')
plt.axhline(alpha,color='gray',linestyle='--',label=r'$\alpha=0.05$')
plt.xlabel(r'$\Delta$')
plt.ylabel("Rejection Rate")
plt.title("Power tests for different values of delta")
plt.legend()
plt.grid(False)
plt.savefig("Power_tests_delta_pooled_vanilla.png", dpi=300, bbox_inches='tight')
plt.show()