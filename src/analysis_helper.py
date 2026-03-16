import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec    
import seaborn as sns
import umap

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.environment import LogNormalConcentration # Adjust import path as needed


@torch.no_grad()
def plot_family_summary(env, physics, receptor_indices, n_points=200):
    """
    Creates a comprehensive summary plot for each ligand family adapted to the discrete model:
    1. Dose-response curves as step functions (Main Frame)
    2. Concentration Distribution (Bottom Frame)
    3. Discrete Binary Assignment / Marginal Probabilities (Right Frame)
    """
    device = env.unit_latent.device
    N_Receptors = receptor_indices.shape[0]
    n_families = env.n_families
    
    # Generate a color palette for the receptors
    colors = plt.cm.viridis(np.linspace(0, 0.9, N_Receptors))
    
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
    
    ax_main = fig.add_subplot(gs[0:3, 0:3])
    ax_bottom = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[0:3, 3], sharey=ax_main)

    for f_idx in range(n_families):
        
        # =====================================================================
        # 1. DATA PREPARATION
        # =====================================================================
        
        # A. Bottom Frame: Exact Concentration PDF
        c_sweep_tensor, c_pdf_tensor = env.get_concentration_sweep(f_idx, n_points=n_points)
        c_sweep_np = c_sweep_tensor.cpu().numpy()
        c_pdf_np = c_pdf_tensor.cpu().numpy()
        
        # Normalize the PDF to act as discrete weights for our exact probability calculation
        c_weights = c_pdf_np / (np.sum(c_pdf_np) + 1e-12)
        
        # B. Main Frame: Dose Response (Sharp Sigmoid / Steps)
        # We don't need method='self_normalized' anymore because it's a true probability [0, 1]
        _, p_o_np = physics.get_dose_response(env, receptor_indices, f_idx, n_points=n_points, method='absolute')
        
        # =====================================================================
        # 2. PLOTTING
        # =====================================================================
        
        # --- Main Frame (Dose Response Steps) ---
        for r in range(N_Receptors):
            # Using standard plot; because of low temperature, it naturally forms a sharp step
            ax_main.plot(c_sweep_np, p_o_np[:, r], color=colors[r], lw=2, label=f"R {r}")
            
        ax_main.set_ylabel("Activity Probability $p(a=1)$", fontsize=9)
        ax_main.tick_params(labelbottom=False, direction='in') 
        ax_main.set_title(f"Receptor Array Binary Response: Ligand Family {f_idx}", fontsize=10, fontweight='bold')
        ax_main.set_ylim(-0.05, 1.05)
        # If your concentration spans orders of magnitude, uncomment the next line!
        ax_main.set_xscale('log')

        # --- Bottom Frame (Concentration Distribution) ---
        ax_bottom.fill_between(c_sweep_np, c_pdf_np, color='gray', alpha=0.4)
        ax_bottom.plot(c_sweep_np, c_pdf_np, color='black', lw=1)
        ax_bottom.set_xlabel("Concentration (M)", fontsize=9)
        ax_bottom.set_ylabel("p(c)", fontsize=9)
        ax_bottom.set_yticks([]) 
        ax_bottom.tick_params(direction='in')
        
        # --- Right Frame (Discrete Binary Assignment) ---
        # Calculate the exact expected marginal probability of firing for each receptor
        # P(a=1) = Sum over all concentrations of: P(fire | c) * P(c)
        p_active = np.sum(p_o_np * c_weights[:, None], axis=0) # Shape: (N_Receptors,)
        p_inactive = 1.0 - p_active
        
        # Plot Grouped Horizontal Bars at y=0.0 (Inactive) and y=1.0 (Active)
        bar_height = 0.6 / N_Receptors # Dynamic scaling so bars don't overlap
        
        for r in range(N_Receptors):
            # Offset each receptor slightly so they stack neatly
            y_offset = (r - N_Receptors/2) * bar_height
            
            # Bar for Inactive Bin (y = 0.0)
            ax_right.barh(0.0 + y_offset, p_inactive[r], height=bar_height, color=colors[r], alpha=0.8)
            # Bar for Active Bin (y = 1.0)
            ax_right.barh(1.0 + y_offset, p_active[r], height=bar_height, color=colors[r], alpha=0.8)
            
        ax_right.set_xlabel("Mass", fontsize=9)
        ax_right.tick_params(labelleft=False, direction='in') 
        
        # Draw dotted lines at exactly 0.0 and 1.0 to guide the eye
        ax_right.axhline(0.0, color='black', linestyle='--', linewidth=0.5, zorder=0)
        ax_right.axhline(1.0, color='black', linestyle='--', linewidth=0.5, zorder=0)
        
    axes = (ax_main, ax_bottom, ax_right)
        
    return fig, axes

@torch.no_grad()
def plot_summary(env, physics, receptor_indices, n_points=200):
    """
    Creates a SINGLE comprehensive summary plot for all ligand families.
    """
    device = env.unit_latent.device
    N_Receptors = receptor_indices.shape[0]
    n_families = env.n_families
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, N_Receptors))
    
    # 1. Initialize Figure ONCE
    fig = plt.figure(figsize=(4, 3))
    gs = GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
    
    ax_main = fig.add_subplot(gs[0:3, 0:3])
    ax_bottom = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[0:3, 3], sharey=ax_main)
    
    # 2. Smart X-Axis Scaling
    if isinstance(env.concentration_model, LogNormalConcentration):
        ax_main.set_xscale('log')
    
    # Array to accumulate the global probabilities for the right panel
    global_p_active = np.zeros(N_Receptors)
    
    for f_idx in range(n_families):
        # --- Data Prep ---
        c_sweep_tensor, c_pdf_tensor = env.get_concentration_sweep(f_idx, n_points=n_points)
        c_sweep_np = c_sweep_tensor.cpu().numpy()
        c_pdf_np = c_pdf_tensor.cpu().numpy()
        
        c_weights = c_pdf_np / (np.sum(c_pdf_np) + 1e-12)
        
        _, p_o_np = physics.get_dose_response(env, receptor_indices, f_idx, n_points=n_points, method='absolute')
        
        # --- Plotting Main and Bottom Frames ---
        # (Using alpha to make overlapping lines readable)
        for r in range(N_Receptors):
            ax_main.plot(c_sweep_np, p_o_np[:, r], color=colors[r], lw=2.5, alpha=1.)
            
        ax_bottom.fill_between(c_sweep_np, c_pdf_np, color='gray', alpha=0.15)
        ax_bottom.plot(c_sweep_np, c_pdf_np, color='black', lw=1., alpha=1.)
        
        # Accumulate the expected probability for this family
        family_p_active = np.sum(p_o_np * c_weights[:, None], axis=0)
        global_p_active += (family_p_active / n_families)
        
    # --- Formatting Main & Bottom ---
    ax_main.set_ylabel("Activity Probability $p(a=1)$", fontsize=9)
    ax_main.tick_params(labelbottom=False, direction='in') 
    ax_main.set_title("Global Receptor Array Binary Response", fontsize=9, fontweight='bold')
    #ax_main.set_ylim(-0.05, 1.05)

    
    ax_bottom.set_xlabel("Concentration (M)", fontsize=9)
    ax_bottom.set_ylabel("p(c)", fontsize=9)
    ax_bottom.set_yticks([]) 
    ax_bottom.tick_params(direction='in')
    
    # --- Plotting Right Frame (Global Marginal Probability) ONCE ---
    global_p_inactive = 1.0 - global_p_active
    bar_height = 0.2 / N_Receptors 
    padding = (N_Receptors / 2) * bar_height * 1.3
    ax_main.set_ylim(-padding, 1.0 + padding)
    
    for r in range(N_Receptors):
        r = (N_Receptors-1)-r
        y_offset = (r - N_Receptors/2) * bar_height
        ax_right.barh(0.0 + y_offset, global_p_inactive[r], height=bar_height, edgecolor=colors[r], alpha=0.8,facecolor='none',linewidth=2.5)
        ax_right.barh(1.0 + y_offset, global_p_active[r], height=bar_height, edgecolor=colors[r], alpha=0.8,facecolor='none',linewidth=2.5)
        
    ax_right.set_xlabel("Global Mass", fontsize=9)
    ax_right.tick_params(labelleft=False, direction='in') 
    ax_right.axhline(0.0, color='black', linestyle='--', linewidth=0.5, zorder=0)
    ax_right.axhline(1.0, color='black', linestyle='--', linewidth=0.5, zorder=0)
    
    return fig, (ax_main, ax_bottom, ax_right)

@torch.no_grad()
def evaluate_model(env,physics,receptor_indices,loss_fn,n_samples=2000,k_knn = 5.):
    device = env.interaction_mu.device
    N_Receptors = receptor_indices.shape[0]
    # draw random ligands
    energies,concs,families = env.sample_batch(batch_size = n_samples)
    # compute the activity array
    activity = physics(energies, concs, receptor_indices)

    return loss_fn._compute_soft_histogram_entropy(activity)


@torch.no_grad()
def plot_latent_radar_chart(env, receptor_indices, receptors_to_plot=None, family_names=None):
    """
    Creates a radar chart showing the relative binding strength of 
    fully assembled receptors (heteromers) across all ligand families.
    """
    n_families = env.n_families
    
    # 1. Fetch exact Unit Energies: (n_units, n_families)
    unit_energies = env.interaction_mu.cpu()
    
    # 2. Compute Receptor Energies
    # Indexing yields (N_Receptors, k_sub, n_families)
    # Mean across k_sub yields (N_Receptors, n_families)
    receptor_energies = unit_energies[receptor_indices].mean(dim=1).numpy()
    
    # Select which specific receptors to plot (default to first 5 if not specified to avoid clutter)
    if receptors_to_plot is None:
        receptors_to_plot = list(range(min(5, len(receptor_indices))))
        
    selected_energies = receptor_energies[receptors_to_plot]
    
    # 3. Convert Energy to Affinity Score
    max_energy = np.max(receptor_energies)
    affinity_scores = max_energy - selected_energies 

    # 4. Setup Radar Chart Angles
    if family_names is None:
        family_names = [f"Fam {i}" for i in range(n_families)]
        
    angles = np.linspace(0, 2 * np.pi, n_families, endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(family_names, fontsize=10)
    ax.set_yticks([]) 
    
    colors = plt.cm.tab10.colors 
    
    # Plot each Receptor's polygon
    for i, r_idx in enumerate(receptors_to_plot):
        values = affinity_scores[i].tolist()
        values += values[:1] 
        
        c = colors[i % len(colors)]
        ax.plot(angles, values, linewidth=2, color=c, label=f"Receptor {r_idx}")
        ax.fill(angles, values, color=c, alpha=0.15)
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    #plt.title("Assembled Receptor Affinity Profile", y=1.08, fontweight='bold')
    plt.tight_layout()
    
    return fig, ax

# Note: if UniformNBall is in environment.py, import it:
# from core.environment import UniformNBall

@torch.no_grad()
def plot_latent_umap(env, receptor_indices, n_samples_per_family=1000, random_state=42):
    """
    Projects the N-dimensional chemical latent space into 2D using UMAP.
    Visualizes the families as density gradients (regions), the family centers as circles,
    and the assembled receptors as numeric indices.
    
    Args:
        env: Instantiated LigandEnvironment.
        receptor_indices: Tensor of shape (N_Receptors, k_sub) mapping units to receptors.
        n_samples_per_family: How many points to sample per family to generate the gradient.
    """
    device = env.unit_latent.device
    n_families = env.n_families
    n_receptors = receptor_indices.shape[0]
    
    # =====================================================================
    # 1. EXTRACT CENTERS AND ASSEMBLED RECEPTORS
    # =====================================================================
    v_families = env.family_latent.detach().cpu().numpy()
    
    # Fetch all raw unit vectors: (n_units, latent_dim)
    v_units_tensor = env.unit_latent.detach().cpu()
    
    # Calculate the centroid of each assembled receptor 
    # Shape: (N_Receptors, k_sub, latent_dim) -> mean(dim=1) -> (N_Receptors, latent_dim)
    v_receptors = v_units_tensor[receptor_indices].mean(dim=1).numpy()
    
    # =====================================================================
    # 2. SAMPLE THE LIGAND REGIONS (To generate the gradient)
    # =====================================================================
    sampled_points = []
    sampled_labels = []
    
    for f_idx in range(n_families):
        # Create a batch of exact identical centers
        center = env.family_latent[f_idx:f_idx+1].expand(n_samples_per_family, -1)
        
        # Draw from the exact distribution defined in the environment
        if env.distribution_type == 'gaussian':
            dist = torch.distributions.Normal(loc=center, scale=env.shape_sigma)
            pts = dist.rsample()
        elif env.distribution_type == 'uniform':
            # Assuming UniformNBall is available in your scope
            from core.environment import UniformNBall
            dist = UniformNBall(loc=center, radius=env.shape_sigma, dim=env.latent_dim)
            pts = dist.rsample()
            
        sampled_points.append(pts.cpu().numpy())
        sampled_labels.extend([f_idx] * n_samples_per_family)
        
    sampled_points = np.vstack(sampled_points)
    sampled_labels = np.array(sampled_labels)
    
    # =====================================================================
    # 3. FIT UMAP PROJECTION
    # =====================================================================
    # We fit UMAP on ALL data simultaneously so the topology is consistent
    all_data = np.vstack([v_families, v_receptors, sampled_points])
    
    print("Fitting UMAP... (This may take a few seconds)")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=random_state)
    embedding = reducer.fit_transform(all_data)
    
    # Unpack the embeddings
    emb_families = embedding[:n_families]
    emb_receptors = embedding[n_families : n_families + n_receptors]
    emb_samples = embedding[n_families + n_receptors :]
    
    # =====================================================================
    # 4. PLOTTING
    # =====================================================================
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Generate a distinct color palette
    colors = plt.cm.tab10.colors if n_families <= 10 else plt.cm.viridis(np.linspace(0, 1, n_families))
    
    # Plot the Density Gradients (Regions)
    for f_idx in range(n_families):
        pts = emb_samples[sampled_labels == f_idx]
        c = colors[f_idx % len(colors)]
        
        # Seaborn KDE creates the beautiful topographical contour gradients
        sns.kdeplot(
            x=pts[:, 0], y=pts[:, 1], 
            ax=ax, fill=True, color=c, alpha=0.3, 
            levels=5, thresh=0.05, linewidths=0.5
        )
        
        # Plot the exact family center (now a circle)
        ax.scatter(
            emb_families[f_idx, 0], emb_families[f_idx, 1], 
            marker='o', s=100, color=c, edgecolor='black', linewidth=1.2,
            zorder=4, label=f'Fam {f_idx}' if f_idx < 10 else ""
        )
        
    # Plot the Assembled Receptors as numbered labels
    for r_idx in range(n_receptors):
        ax.text(
            emb_receptors[r_idx, 0], emb_receptors[r_idx, 1], str(r_idx),
            fontsize=8, ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.2', alpha=0.8),
            zorder=5
        )
    
    # Add a dummy scatter point so "Receptors" appears cleanly in the legend
    ax.scatter([], [], marker='o', color='white', edgecolor='black', label='Receptors')
    
    # Clean up aesthetics
    ax.set_title(f"UMAP Projection of {env.latent_dim}D Chemical Latent Space", fontsize=9, fontweight='bold')
    ax.set_xlabel("UMAP Dimension 1", fontsize=9)
    ax.set_ylabel("UMAP Dimension 2", fontsize=9)

    ax.set_xticks([])
    ax.set_yticks([])
    
    # Shrink current axis by 20% to put legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    return fig, ax