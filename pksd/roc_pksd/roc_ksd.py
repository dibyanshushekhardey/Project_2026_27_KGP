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
#@title KSD Test ROC Curve plots

n = 100
num_boot = 100
n_trials = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sigma_list = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] #, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
results = {}
plt.figure(figsize=(6, 6))

summary_results = []
roc_results = []
trial_results = []

for sigma in sigma_list:
  #ksd_test = KSDTest(means=[0.0, 6.0],sigmas=[sigma, sigma],weights=[0.5, 0.5],device=device)
  p_values_null = []
  p_values_alt = []
  ksd_test = KSDTest(means=[-1, 1],sigmas=[sigma, sigma],weights=[0.5, 0.5],device=device)
  for trial in range(n_trials):
    # Null
    p_x0 = sample_gmm(n=n,means=[-6, 6],sigmas=[sigma, sigma],weights=[0.5, 0.5],device=device)
    #p_x0 = torch.randn(n, device=device) * sigma + mu
    scale0 = median_heuristic(p_x0)
    scale0 = torch.clamp(scale0, min=1e-3)
    ksd0, U0 = ksd_test.compute_ksd(p_x0, scale0)

    boot_stats_0 = []
    for _ in range(num_boot):
        boot_stats_0.append(ksd_test.bootstrap_stat(U0))
    boot_stats_tensor0 = torch.stack(boot_stats_0)
    p0 = (boot_stats_tensor0 >= ksd0).float().mean().item()
    p_values_null.append(p0)

    # Alternate
    p_x1 = sample_gmm(n=n,means=[6],sigmas=[sigma],weights=[1],device=device)
    #ksd_test = KSDTest(means=[3],sigmas=[sigma],weights=[1],device=device)
    scale1 = median_heuristic(p_x1)
    scale1 = torch.clamp(scale1, min=1e-3)
    ksd1, U1 = ksd_test.compute_ksd(p_x1, scale1)

    boot_stats_1 = []
    for _ in range(num_boot):
        boot_stats_1.append(ksd_test.bootstrap_stat(U1))
    boot_stats_tensor1 = torch.stack(boot_stats_1)
    p1 = (boot_stats_tensor1 >= ksd1).float().mean().item()
    p_values_alt.append(p1)

  p_null = np.array(p_values_null)
  p_alt = np.array(p_values_alt)
  thresholds = np.linspace(0, 1, 100)

  fpr = [np.mean(p_null < t) for t in thresholds]
  tpr = [np.mean(p_alt < t) for t in thresholds]
  auc_value = np.trapz(tpr, fpr)
  results[sigma] = {
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
plt.title(f"ROC Curves - KSD Bootstrap Test\nn={n}, shift=1.0, num_boot={num_boot}", fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('roc_curve_ksd.png', dpi=300, bbox_inches='tight')
plt.show()

summary_df = pd.DataFrame(summary_results)
roc_df = pd.DataFrame(roc_results)
trial_df = pd.DataFrame(trial_results)

summary_df.to_csv("ksd_summary.csv", index=False)
roc_df.to_csv("ksd_roc.csv", index=False)
trial_df.to_csv("ksd_trial.csv", index=False)