import torch
import math
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions import Categorical
from tqdm import trange
import pandas as pd
import numpy as np
from torchmin.bfgs import _minimize_bfgs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from pksd.ksd import KSDTest, PKSD, OSPKSD, SPKSD
from pksd.functions import sample_gmm
from pksd.functions import median_heuristic

#@title SPKSD ROC Curves
n = 100
num_boot = 100
n_trials = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
jump_ls = torch.tensor([0.5, 1.0, 2.0, 3.0],device=device)
T = 50
num_starts=30
sigma_list = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
results = {}
plt.figure(figsize=(6, 6))

summary_results = []
roc_results = []
trial_results = []

for sigma in sigma_list:
  p_values_null = []
  p_values_alt = []
  spksd = SPKSD(means=[-6, 6], sigmas=[sigma, sigma], weights=[0.5, 0.5],jump_ls=jump_ls,T=T, num_starts=num_starts,device=device,)

  for trial in range(n_trials):
    # Null
    test_null = sample_gmm( n=n,means=[-6, 6],sigmas=[sigma, sigma],weights=[0.5, 0.5],device=device)
    xt_null = spksd.perturb_all(test_null)
    ksd0, U0 = spksd.compute_spksd(xt_null)
    boot_stats_0 = torch.stack([spksd.bootstrap_stat(U0)for _ in range(num_boot)])
    p0 = (boot_stats_0 >= ksd0).float().mean().item()

    # Alt
    test_alt = sample_gmm( n=n,means=[6],sigmas=[sigma],weights=[1],device=device)
    xt_alt = spksd.perturb_all(test_alt)
    ksd1, U1 = spksd.compute_spksd(xt_alt)
    boot_stats_1 = torch.stack([spksd.bootstrap_stat(U1)for _ in range(num_boot)])
    p1 = (boot_stats_1 >= ksd1).float().mean().item()


    p_values_null.append(p0)
    p_values_alt.append(p1)
    #print("="*50)

  p_null = np.array(p_values_null)
  p_alt = np.array(p_values_alt)
  thresholds = np.linspace(0, 1, 100)
  fpr = [np.mean(p_null < t) for t in thresholds]
  tpr = [np.mean(p_alt < t) for t in thresholds]
  auc_value = np.trapz(tpr, fpr)
  results= {
      'fpr': fpr,
      'tpr': tpr,
      'auc': auc_value,
      'p_null_mean': p_null.mean(),
      'p_alt_mean': p_alt.mean()
  }

  plt.plot(fpr, tpr, linewidth=2, label=f'σ = {sigma} (AUC = {auc_value:.3f})') #(AUC = {auc_value:.3f})
  #storing sign=ma, auc, p_null_mean, p_alt_mean
  summary_results.append({
    "sigma": sigma,
    "auc": auc_value,
    "p_null_mean": p_null.mean(),
    "p_alt_mean": p_alt.mean(),
  })

  # storing the roc results
  for thr, fp, tp in zip(thresholds, fpr, tpr):
    roc_results.append({
        "sigma": sigma,
        "threshold": thr,
        "fpr": fp,
        "tpr": tp,
    })

  # storing trial results
  trial_results.append({
        "sigma": sigma,
        "trial": trial,
        "p_null": p0,
        "p_alt": p1,
    })
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.50)')

# Formatting
plt.xlabel("False Positive Rate (Type I Error)", fontsize=12)
plt.ylabel("True Positive Rate (Power)", fontsize=12)
plt.title(f"ROC Curves - spKSD Bootstrap Test\nn={n}, shift=1.0, num_boot={num_boot}", fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('roc_curve_spKSD.png', dpi=300, bbox_inches='tight')
plt.show()

summary_df = pd.DataFrame(summary_results)
roc_df = pd.DataFrame(roc_results)
trial_df = pd.DataFrame(trial_results)

summary_df.to_csv("spksd_summary.csv", index=False)
roc_df.to_csv("spksd_roc.csv", index=False)
trial_df.to_csv("spksd_trial.csv", index=False)