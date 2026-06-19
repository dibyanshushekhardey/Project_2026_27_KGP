#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:55:03 2026

@author: dibyanshu
"""
import torch
#@title Langevin MCMC Sampling (fixed time step)

def langevin_refine(score_model, x_init, t, num_steps=100, step_size=1e-3, device=device):
  """
  Refine samples using Langevin MCMC driven by a pretrained score network.
  Parameters
  ----------
  score_model : torch.nn.Module
      Trained score network.

  x_init : torch.Tensors
      Initial samples to refine.

  t : float
      Diffusion/noise level at which the score
      network is evaluated.

  num_steps : int
      Number of Langevin iterations.

  step_size : float
      Langevin step size ε.

  device : str or torch.device
      Computation device.

  Returns
  -------
  x : torch.Tensor
      Final refined samples after all
      Langevin updates.

  x_mean : torch.Tensor
      Deterministic component of the final
      update before noise injection.
  """
  x = x_init.clone().to(device)
  batch_size = x.shape[0]
  t_batch = torch.ones(batch_size, device=device) * t
  with torch.no_grad():
    for _ in range(num_steps):
        score = score_model(x, t_batch)
        noise = torch.randn_like(x)
        x_mean = x + step_size * score
        x = (x_mean
            + torch.sqrt(torch.tensor(2 * step_size, device=device)) * noise
        )

  return x, x_mean