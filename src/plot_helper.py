import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

@torch.no_grad()
def plot_family_summary(env, physics, receptor_indices, info_loss, n_samples=2000):
    """
    Creates a comprehensive summary plot for each ligand family, including:
    1. Dose-response curves (Main Frame)
    2. Concentration Distribution (Bottom Frame)
    3. Activity KDE Distribution (Right Frame)
    
    Args:
        env: LigandEnvironment instance
        physics: Receptor instance
        receptor_indices: Tensor of shape (N_Receptors, k_sub)
        info_loss: InformationLoss instance (used for KDE)
        n_samples: Number of random ligands to draw for the KDE estimation
    """
    device = env.interaction_mu.device
    N_Receptors = receptor_indices.shape[0]
    n_families = env.n_families
    
    # Generate a color palette for the receptors
    colors = plt.cm.viridis(np.linspace(0, 0.9, N_Receptors))
    figs,axes = list(),list() # store the figures, and axes generated
    
    for f_idx in range(n_families):
                # =====================================================================
        # 1. DATA PREPARATION (Using built-in methods)
        # =====================================================================
        
        # A. Bottom Frame: Exact Concentration PDF
        c_sweep_tensor, c_pdf_tensor = env.get_concentration_sweep(f_idx, n_points=200)
        c_pdf_x = c_sweep_tensor.cpu().numpy()
        c_pdf_np = c_pdf_tensor.cpu().numpy()
            
        # B. Main Frame: Deterministic Dose-Response Curves
        # c_sweep_np will be exactly identical to c_pdf_x!
        c_sweep_np, p_o_curves_np = physics.get_dose_response(env, receptor_indices, f_idx, n_points=200)
        
        # C. Right Frame: Sampled Activity KDE (Remains unchanged)
        # High-density sampling for a smooth KDE plot
        energies, concentrations = env.sample_specific_family(2000, f_idx)
        activity = physics(energies, concentrations, receptor_indices)
        
        y_query = torch.linspace(0.0, 1.0, 100, device=device)
        query_points = y_query.unsqueeze(1).expand(100, N_Receptors)
        
        activity_density = info_loss._compute_kde(samples=activity, query_points=query_points)
        
        y_query_np = y_query.cpu().numpy()
        activity_density_np = activity_density.cpu().numpy()

        # =====================================================================
        # 2. PLOTTING
        # =====================================================================
        fig = plt.figure(figsize=(6, 4))
        gs = GridSpec(4, 4, figure=fig, wspace=0.1, hspace=0.1)
        
        # Create axes
        ax_main = fig.add_subplot(gs[0:3, 0:3])
        ax_bottom = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[0:3, 3], sharey=ax_main)
        
        # --- Main Frame ---
        for r in range(N_Receptors):
            ax_main.plot(c_sweep_np, p_o_curves_np[:, r], color=colors[r], lw=2, alpha=0.8)
        
        ax_main.set_xscale('log')
        ax_main.set_ylim(-0.02, 1.02)
        ax_main.set_ylabel("Activation ($p_{open}$)", fontsize=9)
        ax_main.tick_params(which='both',labelbottom=False,direction='in') # Hide x-labels to share with bottom frame
        #ax_main.grid(True, which="both", ls="--", alpha=0.3)
        ax_main.set_title(f"Receptor Array Response: Ligand Family {f_idx}", fontsize=9)

        # --- Bottom Frame (Concentration Distribution) ---
        ax_bottom.fill_between(c_sweep_np, c_pdf_np, color='gray', alpha=0.4)
        ax_bottom.plot(c_sweep_np, c_pdf_np, color='black', lw=1)
        ax_bottom.set_xlabel("Concentration (M)", fontsize=9)
        ax_bottom.set_ylabel("p(c)", fontsize=9)
        ax_bottom.set_yticks([]) # Hide y-ticks on density
        ax_bottom.tick_params(direction='in')
        
        # --- Right Frame (Activity KDE) ---
        for r in range(N_Receptors):
            # Notice x and y are swapped: we plot density on X, probability on Y
            ax_right.fill_betweenx(y_query_np, activity_density_np[:, r], 
                                   color=colors[r], alpha=0.2)
            ax_right.plot(activity_density_np[:, r], y_query_np, color=colors[r], lw=1.5)
            
        ax_right.set_xlabel(r"$p(a)$", fontsize=9)
        ax_right.tick_params(labelleft=False,direction='in') # Hide y-labels to share with main frame
        ax_right.set_xticks([]) # Hide x-ticks on density
        
        figs.append(fig)
        axes.append((ax_main,ax_bottom,ax_right))
    return figs, axes

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

@torch.no_grad()
def plot_family_summary(env, physics, receptor_indices, info_loss, n_samples=2000, n_plot_families=None):
    """
    Creates a comprehensive summary plot for each ligand family, including:
    1. Dose-response curves with +/- 2 sigma shaded bounds (Main Frame)
    2. Concentration Distribution (Bottom Frame)
    3. Activity KDE Distribution (Right Frame)
    
    Args:
        env: LigandEnvironment instance
        physics: Receptor instance
        receptor_indices: Tensor of shape (N_Receptors, k_sub)
        info_loss: InformationLoss instance (used for KDE)
        n_samples: Number of random ligands to draw for the KDE estimation
        n_plot_families: Int. If set, randomly samples this many families to plot.
    """
    device = env.interaction_mu.device
    N_Receptors = receptor_indices.shape[0]
    
    # Determine which families to plot
    if n_plot_families is not None and n_plot_families < env.n_families:
        families_to_plot = np.random.choice(env.n_families, size=n_plot_families, replace=False)
        families_to_plot = np.sort(families_to_plot) # Keep them in index order
    else:
        families_to_plot = range(env.n_families)
    
    # Generate a color palette for the receptors
    colors = plt.cm.viridis(np.linspace(0, 0.9, N_Receptors))
    figs, axes = list(), list() 
    
    for f_idx in families_to_plot:
        # =====================================================================
        # 1. DATA PREPARATION
        # =====================================================================
        
        # A. Bottom Frame: Exact Concentration PDF
        c_sweep_tensor, c_pdf_tensor = env.get_concentration_sweep(f_idx, n_points=200)
        c_sweep_np = c_sweep_tensor.cpu().numpy()
        c_pdf_np = c_pdf_tensor.cpu().numpy()
            
        # B. Main Frame: Deterministic Dose-Response Curves (Mean)
        _, p_o_curves_np = physics.get_dose_response(env, receptor_indices, f_idx, n_points=200)
        
        # B.2 Calculate the +/- 2 Sigma Envelope
        mu_family = env.interaction_mu[:, f_idx, :] # (n_units, 2)
        sigma_family = torch.exp(env.interaction_log_sigma[:, f_idx, :]) # (n_units, 2)
        
        mu_receptor = mu_family[receptor_indices]   # (R, k_sub, 2)
        sigma_receptor = sigma_family[receptor_indices] # (R, k_sub, 2)
        
        # Max Activation (Strong Open affinity (-2s), Weak Closed affinity (+2s))
        E_o_min = (mu_receptor[..., 0] - 2 * sigma_receptor[..., 0]).unsqueeze(0).expand(200, N_Receptors, physics.k_sub)
        E_c_max = (mu_receptor[..., 1] + 2 * sigma_receptor[..., 1]).unsqueeze(0).expand(200, N_Receptors, physics.k_sub)
        
        # Min Activation (Weak Open affinity (+2s), Strong Closed affinity (-2s))
        E_o_max = (mu_receptor[..., 0] + 2 * sigma_receptor[..., 0]).unsqueeze(0).expand(200, N_Receptors, physics.k_sub)
        E_c_min = (mu_receptor[..., 1] - 2 * sigma_receptor[..., 1]).unsqueeze(0).expand(200, N_Receptors, physics.k_sub)
        
        c_reshaped = c_sweep_tensor.view(200, 1, 1)
        p_o_max_np = physics.p_open(c_reshaped, E_o_min, E_c_max).cpu().numpy()
        p_o_min_np = physics.p_open(c_reshaped, E_o_max, E_c_min).cpu().numpy()

        # C. Right Frame: Sampled Activity KDE
        energies, concentrations = env.sample_specific_family(n_samples, f_idx)
        activity = physics(energies, concentrations, receptor_indices)
        
        y_query = torch.linspace(0.0, 1.0, 100, device=device)
        query_points = y_query.unsqueeze(1).expand(100, N_Receptors)
        
        activity_density = info_loss._compute_kde(samples=activity, query_points=query_points)
        
        y_query_np = y_query.cpu().numpy()
        activity_density_np = activity_density.cpu().numpy()

        # =====================================================================
        # 2. PLOTTING
        # =====================================================================
        fig = plt.figure(figsize=(6, 4))
        gs = GridSpec(4, 4, figure=fig, wspace=0.1, hspace=0.1)
        
        ax_main = fig.add_subplot(gs[0:3, 0:3])
        ax_bottom = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[0:3, 3], sharey=ax_main)
        
        # --- Main Frame ---
        for r in range(N_Receptors):
            # Plot the mean line
            ax_main.plot(c_sweep_np, p_o_curves_np[:, r], color=colors[r], lw=2, alpha=0.8)
            # Plot the +/- 2 sigma shaded area
            ax_main.fill_between(c_sweep_np, p_o_min_np[:, r], p_o_max_np[:, r], color=colors[r], alpha=0.15)
        
        ax_main.set_xscale('log')
        ax_main.set_ylim(-0.02, 1.02)
        ax_main.set_ylabel("Activation ($p_{open}$)", fontsize=9)
        ax_main.tick_params(which='both', labelbottom=False, direction='in') 
        ax_main.set_title(f"Receptor Array Response: Ligand Family {f_idx}", fontsize=9)

        # --- Bottom Frame (Concentration Distribution) ---
        ax_bottom.fill_between(c_sweep_np, c_pdf_np, color='gray', alpha=0.4)
        ax_bottom.plot(c_sweep_np, c_pdf_np, color='black', lw=1)
        ax_bottom.set_xlabel("Concentration (M)", fontsize=9)
        ax_bottom.set_ylabel("p(c)", fontsize=9)
        ax_bottom.set_yticks([]) 
        ax_bottom.tick_params(direction='in')
        
        # --- Right Frame (Activity KDE) ---
        for r in range(N_Receptors):
            ax_right.fill_betweenx(y_query_np, activity_density_np[:, r], color=colors[r], alpha=0.2)
            ax_right.plot(activity_density_np[:, r], y_query_np, color=colors[r], lw=1.5)
            
        ax_right.set_xlabel(r"$p(a)$", fontsize=9)
        ax_right.tick_params(labelleft=False, direction='in') 
        ax_right.set_xticks([]) 
        
        figs.append(fig)
        axes.append((ax_main, ax_bottom, ax_right))
        
    return figs, axes