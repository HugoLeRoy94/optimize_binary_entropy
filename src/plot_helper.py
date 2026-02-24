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
        # 1. DATA PREPARATION
        # =====================================================================
        
        # A. Main Frame: Deterministic Dose-Response Curves (Using Mean Energies)
        ln_center = env.concentration_model.get_expected_log_c()[f_idx]
        ln_x = torch.linspace(ln_center - 9.2, ln_center + 9.2, 200, device=device)
        c_sweep = torch.exp(ln_x)
        
        # Get mean energies for this family
        mu_family = env.interaction_mu[:, f_idx, :] # (n_units, 2)
        mu_receptor = mu_family[receptor_indices]   # (R, k_sub, 2)
        
        # Expand for batch calculation
        E_open = mu_receptor[..., 0].unsqueeze(0).expand(200, N_Receptors, -1)
        E_closed = mu_receptor[..., 1].unsqueeze(0).expand(200, N_Receptors, -1)
        c_reshaped = c_sweep.view(200, 1, 1)
        
        # Compute probabilities
        p_o_curves = physics.p_open(c_reshaped, E_open, E_closed) # (200, R)
        
        # B. Bottom Frame: Exact Concentration PDF
        d = env.concentration_model.get_distribution(f_idx)
        if env.concentration_model.__class__.__name__ == "LogNormalConcentration":
            # PDF is evaluated in log10 space
            log10_c = ln_x / math.log(10.0)
            c_pdf = torch.exp(d.log_prob(log10_c))
        else:
            # PDF is evaluated in linear space
            c_pdf = torch.exp(d.log_prob(c_sweep))
            
        # C. Right Frame: Sampled Activity KDE
        # Sample realistic environment data for this specific family
        family_ids = torch.full((n_samples,), f_idx, dtype=torch.long, device=device)
        concentrations = env.concentration_model.sample(n_samples, family_ids)
        
        # Sample interaction energies incorporating environment standard deviations
        mu_T = env.interaction_mu.permute(1, 0, 2)
        sigma_T = torch.exp(env.interaction_log_sigma.permute(1, 0, 2))
        batch_mus = mu_T[family_ids]
        batch_sigmas = sigma_T[family_ids]
        energies = torch.distributions.Normal(batch_mus, batch_sigmas).rsample()
        
        # Forward pass to get activities
        activity = physics(energies, concentrations, receptor_indices) # (n_samples, R)
        
        # Use InformationLoss to compute KDE
        y_query = torch.linspace(0.0, 1.0, 100, device=device)
        query_points = y_query.unsqueeze(1).expand(100, N_Receptors)
        # density shape: (100, N_Receptors)
        activity_density = info_loss._compute_kde(samples=activity, query_points=query_points)
        
        # Convert tensors to numpy for matplotlib
        c_sweep_np = c_sweep.cpu().numpy()
        p_o_curves_np = p_o_curves.cpu().numpy()
        c_pdf_np = c_pdf.cpu().numpy()
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