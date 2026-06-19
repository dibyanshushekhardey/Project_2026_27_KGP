#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:52:06 2026

@author: dibyanshu
"""
#@title Importing packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import os
from google.colab import drive
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
# import tqdm
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Bandwidth:
  def __init__(self, X=None):
    self.X=X
    self._bandwidth=None

  def median_heuristic(self, X):
    if X.dim() == 1:
      X = X.reshape(-1, 1)
    dists_sq = torch.cdist(X, X, p=2) ** 2
    # remove diagonal zeros
    mask = ~torch.eye(X.shape[0], dtype=torch.bool, device=X.device)
    vals = dists_sq[mask]
    H_n = torch.median(vals)
    h = torch.sqrt(H_n / 2.0)
    return h