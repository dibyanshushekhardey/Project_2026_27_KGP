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

#@title find modes file functions

#@title find modes file functions

def pairwise_mahalanobis(inv_hessian, x):
    x_h = x.unsqueeze(0)
    res = x_h @ inv_hessian @ x_h.t()
    return res[0, 0]

def merge_modes(inv_hessians, end_pts, threshold, log_prob, threshold_ignore=1e-8):
    M = end_pts.shape[0]
    mode_list = [end_pts[0]]
    inv_hessians_list = [inv_hessians[0]]

    for i in range(1, M):
        maha_dist_list = []
        end_pt_i = end_pts[i]
        inv_hess_i = inv_hessians[i]

        for j in range(len(mode_list)):
            inv_hess = 0.5 * (inv_hessians_list[j] + inv_hess_i)
            diff = mode_list[j] - end_pt_i
            maha_dist = pairwise_mahalanobis(inv_hess, diff)
            maha_dist_list.append(maha_dist)

        maha_dist_tensor = torch.stack(maha_dist_list)
        argmin_i = torch.argmin(maha_dist_tensor).item()
        min_maha_dist = maha_dist_tensor[argmin_i]

        if min_maha_dist < threshold:
            closest_mode = mode_list[argmin_i]

            if log_prob(closest_mode.unsqueeze(0)) < log_prob(end_pt_i.unsqueeze(0)):
                mode_list[argmin_i] = end_pt_i
                inv_hessians_list[argmin_i] = inv_hess_i

        elif log_prob(end_pt_i.unsqueeze(0)) > torch.log(
            torch.tensor(threshold_ignore, device=end_pt_i.device)
        ):
            mode_list.append(end_pt_i)
            inv_hessians_list.append(inv_hess_i)

    return mode_list, inv_hessians_list

def run_bfgs(start_pts, log_prob_fn, grad_log=None,verbose=False, max_iterations=100):
    device = start_pts.device
    positions = []
    inv_hessians = []
    converged = []
    for start in start_pts:
        x0 = start.detach().clone()
        def objective(x):
            x_tensor = x.to(dtype=start.dtype, device=device).unsqueeze(0)
            return -log_prob_fn(x_tensor).squeeze()
        if grad_log is None:
            def gradient(x):
              x_tensor = torch.tensor(x,dtype=start.dtype,device=device,requires_grad=True,)
              loss = -log_prob_fn(x_tensor.unsqueeze(0)).sum()
              grad = torch.autograd.grad(loss, x_tensor)[0]
              return grad.detach().cpu().numpy()

        else:
            def gradient(x):
                x_tensor = torch.tensor(x,dtype=start.dtype,device=device,).unsqueeze(0)
                return (-grad_log(x_tensor)).squeeze(0).cpu().numpy()
        
        result = _minimize_bfgs(fun=objective, x0=x0,max_iter=max_iterations,)
        positions.append(result.x.detach().to(dtype=start.dtype, device=device))
        inv_hessians.append(result.hess_inv.detach().to(dtype=start.dtype, device=device))

        converged.append(result.success)

    if verbose and not all(converged):
        print(
            f"{len(converged)-sum(converged)} of {len(converged)} chains did not converge"
        )

    return {
        "position": torch.stack(positions),
        "inverse_hessian_estimate": torch.stack(inv_hessians),
        "converged": torch.tensor(converged, device=device),
    }

def find_modes(start_pts,log_prob_fn,threshold,grad_log=None,threshold_ignore=1e-8,max_iterations=50,):
    bfgs = run_bfgs(
        start_pts,
        log_prob_fn,
        grad_log=grad_log,
        max_iterations=max_iterations,
    )

    end_pts = bfgs["position"]
    inverse_hessian_estimate = bfgs["inverse_hessian_estimate"]

    mode_list, inv_hess_list = merge_modes(
        inverse_hessian_estimate,
        end_pts,
        threshold,
        log_prob_fn,
        threshold_ignore=threshold_ignore,
      )
    # print("Number of modes:", len(mode_list))
    # print("Modes:")
    # for m in mode_list:
    #     print(m.cpu().numpy())  
    return mode_list, inv_hess_list

def pairwise_directions(modes, return_index=False, ordered=True):
    n = len(modes)
    dir_list = []
    index = []

    if ordered:
        for i in range(n):
            for j in range(n):
                if i != j:
                    dir_list.append(modes[i] - modes[j])
                    index.append((i, j))
    else:
        for i in range(n - 1):
            for j in range(i + 1, n):
                dir_list.append(modes[i] - modes[j])
                index.append((i, j))

    if return_index:
        return dir_list, index
    return dir_list