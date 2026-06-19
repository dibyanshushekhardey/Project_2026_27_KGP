#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:52:56 2026

@author: dibyanshu
"""
import torch

#@title KSD Multidimesnsional
class KSDMultiDim:
  def __init__(self, score_model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Initialize the multidimensional KSD test.

    Parameters
    ----------
    score_model : torch.nn.Module
        Trained score network used to estimate
        ∇x log p_t(x).

    device : str
        CPU or GPU device.

    Attributes
    ----------
    current_t : float
        Diffusion time/noise level used when
        evaluating the score network.

    bandwidth_calculator : Bandwidth
        Utility class used to compute kernel
        bandwidths via the median heuristic.
    """
    self.score_model = score_model
    self.device = device
    self.current_t = None
    self.bandwidth_calculator = Bandwidth()

  def neural_score(self, X_flat, t=None):
    """
    Evaluate the neural score model.

    Parameters
    ----------
    X_flat : torch.Tensor
        Flattened images of shape (n,d).

    t : float
        Diffusion time level.

    Returns
    -------
    score : torch.Tensor
        Estimated score vectors

            ∇x log p_t(x)

        of shape (n,d).
        """
    if t is None:
      raise ValueError("This is a multidimensional test, so value of t must be provided")
    n = X_flat.shape[0]
    X_img = X_flat.reshape(-1, 1, 28, 28)
    t_batch = torch.ones(n, device=X_flat.device, dtype=X_flat.dtype) * t
    with torch.no_grad():
      score = score_model(X_img, t_batch)
    return score.reshape(n, -1)

  def score_function(self, x):
    """
    Compute score vectors for a batch of samples.

    Parameters
    ----------
    x : torch.Tensor
        Input samples.

    Returns
    -------
    score : torch.Tensor
        Neural score evaluations.
    """
    x_flat = x.to(self.device)
    if self.current_t is None:
      raise ValueError("Set self.current_t before calling score_function")
    t = torch.ones(x_flat.shape[0], device=self.device, dtype=x.dtype) * self.current_t
    if x.dim() > 2:
      score = self.neural_score(x.reshape(x.shape[0], -1), t)
    else:
      score = self.neural_score(x, t)
    return score

  def rbf_kernel(self, x1, x2, scale):
    """
    Multivariate Gaussian RBF kernel

        k(x,y)
        =
        exp(
            -||x-y||²/(2h²)
        )

    Parameters
    ----------
    x1, x2 : torch.Tensor
        Input sample pairs.

    scale : torch.Tensor
        Kernel bandwidth.

    Returns
    -------
    k : torch.Tensor
        Pairwise kernel matrix.
    """
    x1 = x1.to(self.device)
    x2 = x2.to(self.device)
    scale = scale.to(self.device)
    diff = x1 - x2
    return torch.exp(-(diff ** 2).sum(dim=-1) / (2 * scale ** 2))

  def grad_x1_rbf(self, x1, x2, scale):
    """
    Gradient of RBF kernel with respect
    to the first argument.
    """
    k = self.rbf_kernel(x1, x2, scale)
    return -k.unsqueeze(-1) * (x1 - x2) / (scale ** 2)

  def grad_x2_rbf(self, x1, x2, scale):
    """
    Gradient of RBF kernel with respect
    to the second argument.
    """
    k = self.rbf_kernel(x1, x2, scale)
    return k.unsqueeze(-1) * (x1 - x2) / (scale ** 2)

  def grad_x1_x2_rbf(self, x1, x2, scale):
    """
    Trace of the mixed Hessian for the multivariate RBF kernel.

    Formula: where d is the data dimension.
    """
    k = self.rbf_kernel(x1, x2, scale)
    diff = x1 - x2
    d = diff.shape[-1]
    squared_dist = (diff ** 2).sum(dim=-1)
    return (d / (scale ** 2) - squared_dist / (scale ** 4)) * k

  def compute_ksd(self, samples, t, h=None):
    """
    Compute the multidimensional
    Kernel Stein Discrepancy.

    Parameters
    ----------
    samples : torch.Tensor
        Input samples of shape (n,d)
        or image tensors.

    t : float
        Diffusion time used when evaluating
        the neural score model.

    h : float, optional
        Kernel bandwidth.
        If omitted, the median heuristic
        is used.

    Returns
    -------
    ksd : float
        Estimated KSD statistic.

    U : torch.Tensor
        Stein kernel matrix.
    """
    samples = samples.to(self.device)
    n = samples.shape[0]

    if samples.dim() > 2:
      samples = samples.reshape(n, -1)
    else:
      samples = samples

    if h is None:
      h = self.bandwidth_calculator.median_heuristic(samples)
    scale = torch.tensor(h, device=self.device)

    self.current_t = t
    x1 = samples.unsqueeze(1)
    x2 = samples.unsqueeze(0)

    k = self.rbf_kernel(x1, x2, scale)
    grad_k_x2 = self.grad_x2_rbf(x1, x2, scale)
    grad_k_x1 = self.grad_x1_rbf(x1, x2, scale)
    hessian_trace = self.grad_x1_x2_rbf(x1, x2, scale)

    S = self.score_function(samples)
    S1 = S.unsqueeze(1)
    S2 = S.unsqueeze(0)

    term1 = (S1 * S2).sum(dim=-1) * k
    term2 = (S1 * grad_k_x2).sum(dim=-1)
    term3 = (S2 * grad_k_x1).sum(dim=-1)

    U = term1 + term2 + term3 + hessian_trace
    U.fill_diagonal_(0)

    ksd = U.sum() / (n * (n - 1))
    return ksd.item(), U

  def bootstrap_stat(self, U):
    """
    Generate one bootstrap realization
    of the KSD test statistic.

    Parameters
    ----------
    U : torch.Tensor
        Stein kernel matrix.

    Returns
    -------
    statistic : torch.Tensor
        Bootstrap statistic used to
        approximate the null distribution.
    """
    n = U.shape[0]
    probs = torch.ones(n, device=self.device) / n
    samples = torch.multinomial(probs, num_samples=n, replacement=True)
    w = torch.bincount(samples, minlength=n).float()

    w_centered = (w / n) - (1.0 / n)

    S = w_centered @ U @ w_centered
    diag_correction = torch.sum(w_centered ** 2 * torch.diag(U))

    return S - diag_correction
