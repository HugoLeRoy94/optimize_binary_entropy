import sys
sys.path.append('/app')
# unit_test/test_single_receptor.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

from src.environment import *
from src.physics import *
from objectives.loss import ProxyInformationLoss,ExactInformationLoss
from src.geometry import generate_receptor_indices
from src.IO import ExperimentLogger

# output class

checkpoints = False
CONF = {
        "n_units": 2,
        "n_families": 100,
        "k_sub": 5,
        "batch_size": 2**12,
        "epochs": 10000,
        "lr": 0.05,
        "k_knn":5,
        "bandwidth_factor": 1.06
    }
logger = ExperimentLogger(base_path="/app/data/",experiment_name="batch_"+str(CONF["batch_size"]))
logger.save_config(CONF)

# 1. SETUP
# -----------------------------------------------------
# We use 1 Unit, 1 Family. The receptor is a homopentamer (Unit 0 five times).

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Modules
conc_strategy = LogNormalConcentration(n_families=CONF['n_families'], init_mean=5.0)
env = LigandEnvironment(CONF['n_units'], CONF['n_families'], conc_model=conc_strategy).to(device)

physics = Receptor(CONF["n_units"], CONF["k_sub"]).to(device)
loss_fn = ExactInformationLoss(k_knn=CONF['k_knn']) # Default bandwidth

# Create the receptor identity: [[0, 0, 0, 0, 0]]
#receptor_indices = torch.zeros(1, CONF["k_sub"], dtype=torch.long, device=device)
#receptor_indices = generate_receptor_indices(n_units=CONF['n_units'],k_sub= CONF['k_sub'],n_sensors=2)
receptor_indices = torch.tensor([[0,0,0,0,0],[1,1,1,1,1]],dtype=torch.long)

# Optimizer
optimizer = optim.Adam(list(env.parameters()) + list(physics.parameters()), lr=CONF["lr"])

# 3. OPTIMIZATION LOOP
# -----------------------------------------------------
print(f"Training for {CONF['epochs']} epochs...")
best_so_far = -float('inf')
logger.save_checkpoint(0, env, physics, receptor_indices, is_best=False)
for epoch in range(CONF['epochs']):
    optimizer.zero_grad()
    
    # A. Sample Batch
    # energies: (B, R, 2), concs: (B,)
    energies, concs, _ = env.sample_batch(CONF['batch_size'])
    
    # B. Physics
    # activity: (B, R)
    activity = physics(energies, concs, receptor_indices)
    

    # C. Loss (Maximize Entropy)
    loss = loss_fn(activity)
    
    loss.backward()
    optimizer.step()
    
    
    if epoch % (CONF['epochs']//100) == 0:       
        stats = loss_fn.make_stats(activity,
                                    bandwidth_factor=CONF['bandwidth_factor'],
                                    k_knn=CONF['k_knn'])
        logger.save_stats(epoch=epoch,stats=stats)
        is_best = (stats["full_array_entropy"] > best_so_far)
        if is_best: 
            best_so_far = stats["full_array_entropy"]
            logger.save_checkpoint(epoch, env, physics, receptor_indices, is_best=is_best)
        elif checkpoints: 
            logger.save_checkpoint(epoch, env, physics, receptor_indices, is_best=is_best)
        print(f"Epoch {epoch}: Entropy = {stats['full_array_entropy']:.4f}")
        

print(f"\nOptimization complete. Results saved to {logger.stats_path}")