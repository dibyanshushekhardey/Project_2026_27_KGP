#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:47:35 2026

@author: dibyanshu
"""
from skimage.metrics import structural_similarity as ssim
import math

def psnr(x_true, x_pred, max_val=1.0):
    mse = torch.mean((x_true - x_pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse.item()))


def compute_ssim(x_true, x_pred):

    x_true = x_true.detach().squeeze().cpu().numpy()
    #x_pred = x_pred.squeeze().cpu().numpy()
    x_pred = x_pred.detach().squeeze().cpu().numpy()
    return ssim(x_true,x_pred,data_range=x_true.max() - x_true.min())