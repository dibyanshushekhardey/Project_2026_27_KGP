import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions import Categorical
from tqdm import trange
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from pksd.ksd import KSDTest, PKSD, OSPKSD, SPKSD
from pksd.functions import sample_gmm

sigma = 1
n = 200
num_boot = 800
delta_list = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
n_trials = 100
rejection_rates_ksd = []
rejection_rates_pksd = []
rejection_rates_ospksd = []
rejection_rates_spksd = []
device = "cuda" if torch.cuda.is_available() else "cpu"

jump_ls = torch.linspace(0.5, 1.5, 21, device=device)
T = 10
num_starts = 30

for delta in delta_list:
  # KSD
  p_values_ksd = []
  # Target distribution q
  ksd_test = KSDTest(means=[-delta/2, delta/2],sigmas=[sigma, sigma], weights=[0.5, 0.5],device=device,)
  for trial in range(n_trials):
      # Sample distribution p (ratio_s = 1.0)
      x_ksd = sample_gmm(n=n,means=[-delta/2, delta/2],sigmas=[sigma, sigma], weights=[0.0, 1.0],device=device,)      # all samples from right mode
      p_ksd = ksd_test.compute_p_value_ksd(x_ksd,num_boot,)
      p_values_ksd.append(p_ksd)

  rejection_rates_ksd.append(np.mean(np.array(p_values_ksd) < 0.05))

  #PKSD
  p_values_pksd = []
  # Target distribution q
  pksd_test = PKSD(means=[-delta/2, delta/2],sigmas=[sigma, sigma], weights=[0.5, 0.5], jump_ls=jump_ls,T=T,device=device,) # target distribution
  for trial in range(n_trials):
      # Sample distribution p (ratio_s = 1.0)
      x_pksd = sample_gmm(n=n,means=[-delta/2, delta/2],sigmas=[sigma, sigma], weights=[0.0, 1.0],device=device,)      # all samples from right mode
      train_pksd = x_pksd[:n//2]
      test_pksd  = x_pksd[n//2:]
      p_pksd = pksd_test.compute_p_value_pksd(x_train=train_pksd, x_test=test_pksd,num_boot=num_boot,)
      p_values_pksd.append(p_pksd)

  rejection_rates_pksd.append(np.mean(np.array(p_values_pksd) < 0.05))

  #OSPKSD
  p_values_ospksd = []
  # Target distribution q
  ospksd_test = OSPKSD(means=[-delta/2, delta/2],sigmas=[sigma, sigma], weights=[0.5, 0.5], jump_ls=jump_ls,T=T,device=device,) # target distribution q
  for trial in range(n_trials):
      # Sample distribution p (ratio_s = 1.0)
      x_ospksd = sample_gmm(n=n,means=[-delta/2, delta/2],sigmas=[sigma, sigma], weights=[0.0, 1.0],device=device,)      # all samples from right mode
      train_ospksd = x_ospksd[:n//2]
      test_ospksd  = x_ospksd[n//2:]
      p_ospksd = ospksd_test.compute_p_value_ospksd(x_train=train_ospksd, x_test=test_ospksd,num_boot=num_boot,)
      p_values_ospksd.append(p_ospksd)

  rejection_rates_ospksd.append(np.mean(np.array(p_values_ospksd) < 0.05))

  #SPKSD
  p_values_spksd = []
  # Target distribution q
  spksd_test = SPKSD(means=[-delta/2, delta/2],sigmas=[sigma, sigma],weights=[0.5, 0.5],jump_ls=jump_ls,T=T,device=device,) # target distribution Q
  for trial in range(n_trials):
      # Sample distribution p (ratio_s = 1.0)
      x_spksd = sample_gmm(n=n,means=[-delta/2, delta/2],sigmas=[sigma, sigma], weights=[0.0, 1.0],device=device,)      # all samples from right mode
      p_spksd = spksd_test.compute_p_value_spksd(x_spksd,num_boot,)
      p_values_spksd.append(p_spksd)

  rejection_rates_spksd.append(np.mean(np.array(p_values_spksd) < 0.05))

plt.figure(figsize=(6,4))

plt.plot(delta_list,rejection_rates_ksd,marker='o',linewidth=2,label='KSD')
plt.plot(delta_list,rejection_rates_pksd,marker='x',linewidth=2,label='PKSD')
plt.plot(delta_list,rejection_rates_ospksd,marker='d',linewidth=2,label='OSPKSD')
plt.plot(delta_list,rejection_rates_spksd,marker='h',linewidth=2,label='SPKSD')
plt.axhline(0.05,color='gray',linestyle='--',label=r'$\alpha=0.05$')
plt.xlabel(r'$\Delta$')
plt.ylabel("Rejection Rate")
plt.legend()
plt.grid(False)

# Save the figure
plt.savefig("rejection_rates.png", dpi=300, bbox_inches="tight")

plt.show()