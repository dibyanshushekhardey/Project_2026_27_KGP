#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:46:49 2026

@author: dibyanshu
"""

# Measurement operators

class GaussianDenoisingOperator:
    def forward(self, x):
        return x

class InpaintingOperator:
    def __init__(self, mask):
        self.mask = mask

    def forward(self, x):
        return self.mask * x