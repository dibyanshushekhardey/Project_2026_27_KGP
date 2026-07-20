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
Run the following command. This commands runs the script .py file in the folders to test the power of the tets statistic compared with pooled and vanilla ksd for different values of delta

```bash
!python -m experiments.power_plots_pooled_ksd.power_plot_vanilla_pooled_ksd
```

Run the following command. This commands runs the script .py file in the folders to test the power of the tets statistic compared with pooled and vanilla ksd for different values of weights for a fixed delta.
```bash
!python -m experiments.power_plots_pooled_ksd.power_weights_compare
```

Also run the 
Papers that were used to produce the results:
Using Using Perturbation to Improve Goodness-of-Fit Tests based on Kernelized Stein Discrepancy - Xing Liu, Andrew B. Duncan, Axel Gandy - https://arxiv.org/abs/2304.14762 Official Code - https://github.com/XingLLiu/pksd

A Kernelized Stein Discrepancy for Goodness-of-fit Tests and Model Evaluation - Qiang Liu, Jason D. Lee, Michael I. Jordan - https://arxiv.org/abs/1602.03253