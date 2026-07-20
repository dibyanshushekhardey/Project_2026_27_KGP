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
from ksd_pooled.ksd_1D_Gaussian.functions import median_heuristic, sample_gmm

#@title Power Plot Unperturbed KSD
sigma = 1
n = 1000
num_boot = 500
delta_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #
n_trials = 100
rejection_rates_ksd = []

for delta in delta_list:
  #KSD
  p_values_ksd = []
  ksd_test = KSDTest(means=[0, delta],sigmas=[sigma, sigma], weights=[0.5, 0.5],device=device,)  # Target distribution p
  for trial in range(n_trials):
      # Sample distribution p (ratio_s = 1.0)
      x_ksd = sample_gmm(n=n,means=[0, delta],sigmas=[sigma, sigma], weights=[1.0, 0],device=device,) #all samples from right mode
      p_ksd = ksd_test.compute_p_value_ksd(x_ksd,num_boot,)
      p_values_ksd.append(p_ksd)

  rejection_rates_ksd.append(np.mean(np.array(p_values_ksd) < 0.05))

plt.figure(figsize=(6,4))
plt.plot(delta_list,rejection_rates_ksd,marker='o',linewidth=2,label='KSD')
plt.axhline(0.05,color='gray',linestyle='--',label=r'$\alpha=0.05$')
plt.xlabel(r'$\Delta$')
plt.ylabel("Rejection Rate")
plt.legend()
plt.grid(False)
plt.savefig("power_plot_vanilla_ksd.png", dpi=300, bbox_inches='tight')
plt.show()