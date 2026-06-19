#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:44:50 2026

@author: dibyanshu
"""

#@title Tweedie Estimate
def compute_x0_hat(x_t, t, score):
    # Tweedie's formula
    sigma_t = marginal_prob_std_fn(t)
    sigma_t = sigma_t[:,None,None,None]
    return x_t + sigma_t**2 * score

#@title Measurement Loss
def measurement_loss(y, x0_hat, operator):
    y_hat = operator.forward(x0_hat)
    return ((y - y_hat)**2).sum()

#@title Gradient Calculation
def compute_gradient(x_t, y, operator, t):
    x_t.requires_grad_(True)
    score = score_model(x_t, t)
    x0_hat = compute_x0_hat(x_t,t, score)
    y_hat = operator.forward(x0_hat)
    loss = ((y - y_hat)**2).sum()
    grad = torch.autograd.grad(loss,x_t)[0]

    return grad, x0_hat

#@title Step size calculation
def compute_step_size(y,x0_hat,operator,zeta_prime):
    residual = y - operator.forward(x0_hat)
    return zeta_prime / (residual.norm() + 1e-8)

def reverse_sde_step(x_t, score, t, dt):
    g = diffusion_coeff_fn(t)
    if g.ndim == 0:
        g = g.view(1)
    g = g[:, None, None, None]
    noise = torch.randn_like(x_t)
    x_mean = x_t + g**2 * score * dt
    x = x_mean + g * torch.sqrt(dt) * noise
    return x, x_mean

def dps_sampler(y,operator, num_steps=1000,zeta_prime=0.01):
    #torch.manual_seed(42)
    t_init = torch.ones(y.shape[0],device=device)
    init_std = marginal_prob_std_fn(t_init)

    init_std = init_std[:,None,None,None].to(y.device)

    x_t = torch.randn_like(y) * init_std
    time_steps = torch.linspace(1.0,1e-3,num_steps,device=device)
    dt = time_steps[0] - time_steps[1]

    for step, t in enumerate(time_steps):
        t_batch = torch.ones(x_t.shape[0],device=device) * t
        score = score_model(x_t,t_batch)

        x_prime,_ = reverse_sde_step(x_t,score,t,dt)
        grad, x0_hat = compute_gradient( x_t,y, operator,t_batch)

        step_size = compute_step_size(y,x0_hat,operator,zeta_prime).to(device)
        x_t = (x_prime- step_size * grad).detach()


    return x0_hat