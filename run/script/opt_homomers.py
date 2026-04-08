import sys
sys.path.append('/app')
# unit_test/test_single_receptor.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import cycle

from src import (generate_receptor_indices,
                generate_cascading_receptors,
                generate_targeted_receptors,
                plot_family_summary,                
                plot_latent_radar_chart,
                evaluate_model,
                plot_summary,
                plot_latent_umap,
                receptor_distances,
                full_array_entropy,
                mean_receptor_distance)
from run import initialize,train,test
from src.IO import ExperimentLogger

n_units=5
k_sub = 5
n_families = 1
N_train = 2**17
N_test = 2**14
CONF = {
        "n_units": n_units,
        "n_families": n_families,
        "latent_dim": 3,
        "k_sub": k_sub,
        "batch_size": N_train,
        "epochs": 5000,
        "lr": 0.05,
        "cov_weight":10.,
        "n_bins":2,
        "bin_temp":0.05,
        "receptor_indices":torch.tensor([[i for _ in range(k_sub)] for i in range(n_units)],dtype=torch.long), # size must be smaller or equal to n_units
        "init_means":[np.random.randint(1,8) for _ in range(n_families)], # size must be n_families
        "shape_sigma": 1.,
        "tolerant":False, # set whether we want tolerance for heteromers on the covariance loss
        "optimizer":"Adam",
        "momentum":0.9,
        "exact_loss":True,
        "temperature":0.1
    }

env, rec, loss_fn, optimize = initialize(CONF, SymmetricEnv=False)

logger = ExperimentLogger(base_path="/app/data/", experiment_name="opt_homomers")
logger.save_config(CONF)

train_out = train(CONF, env, rec, loss_fn, optimize,measurement_fns=[full_array_entropy,mean_receptor_distance,])

# Save training statistics 
epochs_run = len(next(iter(train_out.values())))
for i in range(epochs_run):
    stats = {k: v[i] for k, v in train_out.items()}
    logger.save_stats(i, stats)

# Save the final checkpoint (best model)
logger.save_checkpoint(
    epoch=CONF["epochs"], env=env, physics=rec, 
    receptor_indices=CONF["receptor_indices"], is_best=True
)