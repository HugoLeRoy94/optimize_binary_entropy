import sys
sys.path.append('/app')
# unit_test/test_single_receptor.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect
from itertools import cycle


from src import (LigandEnvironment,
                SymmetricLigandEnvironment,
                BinaryReceptor,
                BaseReceptor,
                generate_receptor_indices, 
                plot_family_summary,
                LogNormalConcentration,
                plot_latent_radar_chart,
                evaluate_model,
                plot_summary,
                plot_latent_umap,
                marginal_entropy,
                full_array_entropy,
                total_correlation)
from objectives import DiscreteProxyLoss,TolerantDiscreteProxyLoss,DiscreteExactLoss


def initialize(CONF:dict,SymmetricEnv=False)->tuple[LigandEnvironment,BaseReceptor,nn.Module,optim.Optimizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conc_strategy = LogNormalConcentration(n_families=CONF['n_families'], 
                                            init_mean=CONF['init_means'])
    if SymmetricEnv:
        env = SymmetricLigandEnvironment(CONF['n_units'],
                            CONF['n_families'], 
                            conc_model=conc_strategy,
                            latent_dim=CONF['latent_dim'],
                            shape_sigma=CONF.get('shape_sigma', 0.5)).to(device)
    else:
        env = LigandEnvironment(CONF['n_units'],
                            CONF['n_families'], 
                            conc_model=conc_strategy,
                            latent_dim=CONF['latent_dim'],
                            shape_sigma=CONF.get('shape_sigma', 0.5)).to(device)
    physics = BinaryReceptor(CONF["n_units"], CONF["k_sub"],temperature=CONF["temperature"]).to(device)
    
    if CONF.get("exact_loss", False):
        loss_fn = DiscreteExactLoss(n_bins=CONF.get('n_bins', 2),
                                    bin_temp=CONF.get("bin_temp", 0.05)).to(device)
    elif CONF.get("tolerant", False):
        loss_fn = TolerantDiscreteProxyLoss(env=env,
                                            receptor_indices=CONF["receptor_indices"],
                                            n_units=CONF["n_units"],
                                            cov_weight = CONF["cov_weight"],
                                            n_bins=CONF['n_bins'],
                                            bin_temp=CONF["bin_temp"]).to(device)
    else:
        loss_fn = DiscreteProxyLoss(cov_weight = CONF["cov_weight"],
                                    n_bins=CONF['n_bins'],
                                    bin_temp=CONF["bin_temp"]).to(device)
    if CONF['optimizer'] == "Adam":
        optimizer = optim.Adam(list(env.parameters()) + 
                                list(physics.parameters()),
                                lr=CONF["lr"])
    elif CONF['optimizer'] == "SGD":
        optimizer = optim.SGD(list(env.parameters()) + 
                                list(physics.parameters()),
                                lr = CONF['lr'],
                                momentum=CONF['momentum'])
    
    return env,physics,loss_fn,optimizer


def train(CONF:dict,
        env:LigandEnvironment,
        physics:BaseReceptor,
        loss_fn:nn.Module,
        optimizer:optim.Optimizer,
        measurement_fns:list=None)->list:

    #env,physics,loss_fn,optimizer = initialize(CONF)

    print(f"Training for {CONF['epochs']} epochs...")

#    if measurement_fns is None:
#        measurement_fns = [full_array_entropy, marginal_entropy, total_correlation]

    scheduler = None
    if CONF.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONF['epochs'], eta_min=1e-5)

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
        if scheduler is not None:
            scheduler.step()
            
        if epoch % (CONF['epochs']//100) == 0:
            with torch.no_grad():
                # 1. Generate a massive evaluation batch
                E_open_stats, concs_stats, _ = env.sample_batch(batch_size=1_000_000)
                
                # 2. Get probabilities
                activity_stats = physics(E_open_stats, concs_stats, CONF["receptor_indices"])
                
                stat = {}
                
                # 3. Measurements
                for fn in measurement_fns:
                    sig = inspect.signature(fn)
                    kwargs = {}
                    # Automatically map the requested arguments
                    if 'env' in sig.parameters: kwargs['env'] = env
                    if 'physics' in sig.parameters: kwargs['physics'] = physics
                    if 'receptor_indices' in sig.parameters: kwargs['receptor_indices'] = CONF["receptor_indices"]
                    if 'loss_fn' in sig.parameters: kwargs['loss_fn'] = loss_fn
                    if 'activity' in sig.parameters: kwargs['activity'] = activity_stats
                    if 'epoch' in sig.parameters: kwargs['epoch'] = epoch
                    
                    result = fn(**kwargs)
                    if isinstance(result, dict):
                        stat.update(result)
                    else:
                        name = getattr(fn, '__name__', str(fn))
                        stat[name] = result
                stat['lr'] = optimizer.param_groups[0]['lr']
                
                stats.append(stat)
    stats = {key:[stats[i][key] for i in range(stats.__len__())] for key in stats[0].keys()}
    return stats

def test(CONF:dict,
    env:LigandEnvironment,
    physics:BaseReceptor,
    loss_fn:nn.Module,
    optimizer:optim.Optimizer,
    indices:torch.Tensor,
    N_samples:int,
    epoch:int = 100)->list:        
    ents = []
    for _ in range(epoch):
        val = evaluate_model(env=env,physics=physics,receptor_indices=indices,loss_fn=loss_fn,n_samples=N_samples)
        ents.append(val)
    return ents