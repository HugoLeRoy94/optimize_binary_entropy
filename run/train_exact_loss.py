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
from src.plot_helper import *

CONF = {
        "n_units": 2,
        "n_families": 2,
        "k_sub": 5,
        "batch_size": 512,
        "epochs": 600,
        "lr": 0.05,
        "cov_weight":10.
    }

# 1. SETUP
# -----------------------------------------------------
# We use 1 Unit, 1 Family. The receptor is a homopentamer (Unit 0 five times).

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Modules
conc_strategy = LogNormalConcentration(n_families=CONF['n_families'], init_mean=5.0)
env = LigandEnvironment(CONF['n_units'], CONF['n_families'], conc_model=conc_strategy).to(device)

physics = Receptor(CONF["n_units"], CONF["k_sub"]).to(device)
loss_fn = ProxyInformationLoss(cov_weight=CONF['cov_weight']) # Default bandwidth

# Create the receptor identity: [[0, 0, 0, 0, 0]]
#receptor_indices = torch.zeros(1, CONF["k_sub"], dtype=torch.long, device=device)
#receptor_indices = generate_receptor_indices(n_units=CONF['n_units'],k_sub= CONF['k_sub'],n_sensors=2)
receptor_indices = torch.tensor([[0,0,0,0,0],[1,1,1,1,1]],dtype=torch.long)
print(receptor_indices)

# Optimizer
optimizer = optim.Adam(list(env.parameters()) + list(physics.parameters()), lr=CONF["lr"])

# 3. OPTIMIZATION LOOP
# -----------------------------------------------------
print(f"Training for {CONF['epochs']} epochs...")

stats = []
for epoch in range(CONF['epochs']):
    optimizer.zero_grad()
    
    # A. Sample Batch
    # energies: (B, 1, 2), concs: (B,)
    energies, concs, _ = env.sample_batch(CONF['batch_size'])
    
    # B. Physics
    # activity: (B, 1)
    activity = physics(energies, concs, receptor_indices)
    

    # C. Loss (Maximize Entropy)
    loss = loss_fn(activity)
    
    loss.backward()
    optimizer.step()
    
    
    if epoch % 100 == 0:
        stats.append(loss_fn.make_stats(activity))
stats = np.array(stats)
stats = {
            "full_array_entropy":stats[:,0],
            "marginal_entropy":stats[:,1],
            "total_correlation":stats[:,2]
        }
