import torch
import torch.nn as nn

class Receptor(nn.Module):
    """
    The Physics Module (Simplified MWC Model).
    
    This module is now parameter-less. It deterministically maps 
    interaction energies and concentrations to receptor activity, 
    accounting for parameter degeneracy by absorbing the leakiness 
    factor into the open-state affinity.
    """
    def __init__(self, n_units: int, k_sub: int = 5):
        super().__init__()
        self.n_units = n_units
        self.k_sub = k_sub
    def p_open(self, c_reshaped: torch.Tensor, E_open: torch.Tensor, E_closed: torch.Tensor):
            """
            Numerically stable MWC calculation using log-space formulation.
            """
            # 1. Safe log concentration
            ln_c = torch.log(c_reshaped + 1e-12) # Add epsilon to prevent log(0)
            
            # 2. Log-Weight of the Open State
            # ln(c * exp(-E_o)) = ln(c) - E_o
            log_w_open_per_unit = ln_c - E_open
            ln_W_open = torch.sum(log_w_open_per_unit, dim=-1) # (Batch, R)
            
            # 3. Log-Weight of the Closed State
            # ln(1 + c * exp(-E_c)) is mathematically identical to softplus(ln(c) - E_c)
            # softplus(x) = ln(1 + exp(x)). It is highly optimized and overflow-safe!
            log_w_closed_per_unit = torch.nn.functional.softplus(ln_c - E_closed)
            ln_W_closed = torch.sum(log_w_closed_per_unit, dim=-1) # (Batch, R)
            
            # 4. Final Probability
            # p_o = exp(ln_W_open) / (exp(ln_W_open) + exp(ln_W_closed))
            # This simplifies exactly to sigmoid(ln_W_open - ln_W_closed)
            p_o = torch.sigmoid(ln_W_open - ln_W_closed)
            
            return p_o

    def forward(self, 
                energies: torch.Tensor, 
                concentrations: torch.Tensor, 
                receptor_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energies: (Batch, n_units, 2) - Sampled interaction energies [E_open, E_closed]
            concentrations: (Batch,) - Sampled concentrations
            receptor_indices: (N_Receptors, k_sub) - Stoichiometry definitions
                
        Returns:
            normalized_activity: (Batch, N_Receptors) 
        """
        batch_size = energies.shape[0]
        n_receptors = receptor_indices.shape[0]
        
        # ----------------------------------------------------------------------
        # A. GATHER ENERGIES FOR SPECIFIC RECEPTORS
        # ----------------------------------------------------------------------
        flat_indices = receptor_indices.view(-1)
        gathered_flat = energies[:, flat_indices, :]
        
        # Shape: (Batch, R, k_sub, 2)
        energies_k = gathered_flat.view(batch_size, n_receptors, self.k_sub, 2)
        
        # E_open is now the EFFECTIVE energy (incorporating the sloppy epsilon)
        E_open = energies_k[..., 0]   # (Batch, R, k)
        E_closed = energies_k[..., 1] # (Batch, R, k)
        
        # ----------------------------------------------------------------------
        # B. COMPUTE ACTIVITY (Simplified MWC)
        # ----------------------------------------------------------------------

        p_o = self.p_open(concentrations.view(batch_size, 1, 1),E_open,E_closed)        
        
        return torch.clamp(p_o, 0.0, 1.0)

    @torch.no_grad()
    def get_dose_response(self, env, receptor_indices, family_id, n_points=200,method = 'absolute'):
        device = env.interaction_mu.device
        N_Receptors = receptor_indices.shape[0]
        
        # 1. Ask environment for the physical concentration sweep
        c_sweep, _ = env.get_concentration_sweep(family_id, n_points)

        # 2. Get the MEAN interaction energies for this family
        mu_family = env.interaction_mu[:, family_id, :] 
        mu_receptor = mu_family[receptor_indices]

        # Prepare for p_open: needs (Batch, R, k_sub)
        E_open = mu_receptor[..., 0].unsqueeze(0).expand(n_points, N_Receptors, self.k_sub)
        E_closed = mu_receptor[..., 1].unsqueeze(0).expand(n_points, N_Receptors, self.k_sub)
        
        # 3. Compute activity
        c_reshaped = c_sweep.view(n_points, 1, 1)
        p_o = self.p_open(c_reshaped, E_open, E_closed) 
        if method == "self_normalized":
            # Divide by the maximum value observed in this specific batch/sweep
            p_max = torch.max(p_o, dim=0, keepdim=True).values
            p_o =  p_o / (p_max + 1e-8) # Avoid division by zero
        
        return c_sweep.cpu().numpy(), p_o.cpu().numpy()