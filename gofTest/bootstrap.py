#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:53:56 2026

@author: dibyanshu
"""
import torch

#@title Bootstrap for calcluating p values

class BootstrapKSD:
  def __init__(self, ksd_test_instance, num_bootstrap=100, random_seed=None):
    """
    Initialize the bootstrap procedure for
    the Kernel Stein Discrepancy (KSD) test.

    Parameters
    ----------
    ksd_test_instance : KSDTest
        Instance containing the KSD computation
        and bootstrap statistic methods.

    num_bootstrap : int
        Number of bootstrap replications used
        to approximate the null distribution.

    random_seed : int, optional
        Seed for reproducibility.
    """
    self.ksd_test = ksd_test_instance
    self.num_bootstrap = num_bootstrap
    if random_seed is not None:
      torch.manual_seed(random_seed)
      np.random.seed(random_seed)

  def compute_p_value(self, U, observed_ksd):
    """
    Estimate the p-value of a KSD test statistic
    using bootstrap resampling.

    Procedure
    ---------
    1. Generate bootstrap realizations of the
        KSD statistic from the Stein kernel matrix U.
    2. Approximate the null distribution using
        these bootstrap samples.
    3. Compute the p-value as

            p = P(T_boot >= T_obs)

        estimated empirically from the bootstrap
        distribution.

    Parameters
    ----------
    U : torch.Tensor
        Stein kernel matrix.

    observed_ksd : torch.Tensor
        Observed KSD statistic computed from
        the data.

    Returns
    -------
    p_value : float
        Bootstrap estimate of the p-value.

    bootstrap_stats_tensor : torch.Tensor
        Collection of bootstrap statistics
        approximating the null distribution.
    """
    bootstrap_stats = []
    for _ in range(self.num_bootstrap):
      bootstrap_stats.append(self.ksd_test.bootstrap_stat(U))
    bootstrap_stats_tensor = torch.stack(bootstrap_stats)
    p_value = (bootstrap_stats_tensor >= observed_ksd).float().mean().item()
    return p_value, bootstrap_stats_tensor