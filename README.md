# Project_2026_27_KGP_MTP
M-tech project files for MTP

for pksd experiment pksd_experiment_power.py following are the steps if you are working in google colab. Run the commands in the colab cell.

Run the following command in cell
```bash
git clone https://github.com/dibyanshushekhardey/Project_2026_27_KGP.git
```


Go to the directory
```bash
%cd /content/Project_2026_27_KGP
```
find_modes.py uses the following package, install using the following
```bash
!pip install pytorch-minimize
```
Run the following for executing the pksd_experiment_power.py file.
```bash
!python pksd_experiment_power.py
```

Run the following for executing the pksd_experiment_power_weights.py file.
```bash
!python pksd_experiment_power_weights.py
```

Papers that were used to produce the results:
Using Using Perturbation to Improve Goodness-of-Fit Tests based on Kernelized Stein Discrepancy - Xing Liu, Andrew B. Duncan, Axel Gandy - https://arxiv.org/abs/2304.14762 Official Code - https://github.com/XingLLiu/pksd

A Kernelized Stein Discrepancy for Goodness-of-fit Tests and Model Evaluation - Qiang Liu, Jason D. Lee, Michael I. Jordan - https://arxiv.org/abs/1602.03253