import sys
sys.path.append('/app')
# unit_test/test_single_receptor.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import cycle

from src import (LigandEnvironment, 
                BinaryReceptor,
                BaseReceptor,
                generate_receptor_indices, 
                plot_family_summary,
                LogNormalConcentration,
                plot_latent_radar_chart,
                evaluate_model,
                plot_summary,
                plot_latent_umap)
from objectives import DiscreteProxyLoss


def initialize(CONF:dict)->tuple[LigandEnvironment,BaseReceptor,DiscreteProxyLoss,optim.Optimizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conc_strategy = LogNormalConcentration(n_families=CONF['n_families'], 
                                            init_mean=CONF['init_means'])
    env = LigandEnvironment(CONF['n_units'], 
                        CONF['n_families'], 
                        conc_model=conc_strategy,
                        latent_dim=CONF['latent_dim']).to(device)
    physics = BinaryReceptor(CONF["n_units"], CONF["k_sub"]).to(device)
    
    loss_fn = DiscreteProxyLoss(cov_weight = CONF["cov_weight"],
                                n_bins=CONF['n_bins'],
                                bin_temp=CONF["bin_temp"]).to(device)

    optimizer = optim.Adam(list(env.parameters()) + 
                            list(physics.parameters()),
                            lr=CONF["lr"])
    
    return env,physics,loss_fn,optimizer


def run(CONF:dict,
        env:LigandEnvironment,
        physics:BaseReceptor,
        loss_fn:DiscreteProxyLoss,
        optimizer:optim.Optimizer)->list:

    #env,physics,loss_fn,optimizer = initialize(CONF)

    print(f"Training for {CONF['epochs']} epochs...")

    stats = []
    for epoch in range(CONF['epochs']):
        optimizer.zero_grad()
        
        # A. Sample Batch
        # energies: (B, 1, 2), concs: (B,)
        energies, concs, _ = env.sample_batch(CONF['batch_size'])
        
        # B. Physics
        # activity: (B, 1)
        activity = physics(energies, concs, CONF["receptor_indices"])
        

        # C. Loss (Maximize Entropy)
        loss = loss_fn(activity)
        
        loss.backward()
        optimizer.step()
        if epoch % (CONF['epochs']//100) == 0:
            with torch.no_grad():
                # 1. Generate a massive evaluation batch
                E_open_stats, concs_stats, _ = env.sample_batch(batch_size=1_000_000)
                
                # 2. Get probabilities
                activity_stats = physics(E_open_stats, concs_stats, CONF["receptor_indices"])
                
                # 3. Compute highly accurate stats
                stat = loss_fn.make_stats(activity)
                print(f"Total Correlation: {stat['total_correlation']:.4f}")
                stats.append(stat)
    stats = {key:[stats[i][key] for i in range(stats.__len__())] for key in stats[0].keys()}
    return stats