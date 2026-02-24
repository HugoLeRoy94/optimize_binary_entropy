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
    
    #def p_open(self,c_reshaped:torch.Tensor,E_open: torch.Tensor,E_closed: torch.Tensor):
    #    """
    #    Args:
    #        c_reshaped: (batch, 1,1)
    #        E_open: (batch, R, k_unit)
    #        E_closed: (batch, R, k_unit)
    #    """
    #    
    #    # Numerator: Product( c / Ko_tilde ) = Product( c * exp(-E_open) )
    #    term_open_per_unit = c_reshaped * torch.exp(-E_open)
    #    term_open = torch.prod(term_open_per_unit, dim=-1) # (Batch, R)
    #    
    #    # Denominator Term 2: Product( 1 + c / Kc ) = Product( 1 + c * exp(-E_closed) )
    #    term_closed_per_unit = 1.0 + c_reshaped * torch.exp(-E_closed)
    #    term_closed = torch.prod(term_closed_per_unit, dim=-1) # (Batch, R)
    #    
    #    # Raw probability
    #    p_c = term_open / (term_open + term_closed)
#
    #    # ----------------------------------------------------------------------
    #    # RENORMALIZE
    #    # ----------------------------------------------------------------------
    #    # With the simplified model, p_min (at c=0) is exactly 0.0
    #    # We only need to calculate p_max (as c -> infinity)
    #    
    #    # As c -> inf, term_open ~ c^k * exp(-Sum(E_open))
    #    # As c -> inf, term_closed ~ c^k * exp(-Sum(E_closed))
    #    # p_max = exp(-Sum(E_open)) / [exp(-Sum(E_open)) + exp(-Sum(E_closed))]
    #    # Divided by exp(-Sum(E_open)):
    #    # p_max = 1 / [1 + exp( Sum(E_open) - Sum(E_closed) )]
    #    
    #    delta_E_sum = torch.sum(E_open - E_closed, dim=-1) # (Batch, R)
    #    p_max = 1.0 / (1.0 + torch.exp(delta_E_sum))
    #    
    #    # Normalize (p_min is 0, so it's just p_c / p_max)
    #    # Add epsilon to denominator to prevent division by zero if p_max is effectively 0
    #    normalized = p_c / (p_max + 1e-8)
    #    return normalized

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
    def get_dose_response(self, c_sweep, env, receptor_index, family_id, n_points=100):
        device = env.interaction_mu.device
        
        # We want a log-uniform sweep to make the sigmoid look pretty
        # Let's get the center in natural log space
        d = env.concentration_model.get_distribution(family_id)
        ln_x = torch.linspace(d.mean - 3*d.stddev, d.mean + 3*d.stddev, n_points,device = env.interaction_mu.device)
                
        c_sweep = 10**ln_x
        

        # 2. Get the MEAN interaction energies for this family
        # interaction_mu shape: (n_units, n_families, 2)
        # We take the mean (mu) rather than sampling to get the "characteristic" curve
        mu_family = env.interaction_mu[:, family_id, :] # (n_units, 2)
        
        # Gather energies for the specific stoichiometry
        # mu_receptor shape: (k_sub, 2)
        mu_receptor = mu_family[receptor_index]

        # Prepare for p_open: needs (Batch, R, k, 2)
        # We treat n_points as the batch dimension
        E_open = mu_receptor[:, 0].expand(n_points, 1, self.k_sub)
        E_closed = mu_receptor[:, 1].expand(n_points, 1, self.k_sub)
        # 3. Compute activity
        # p_open expects c_reshaped: (Batch, 1, 1)
        c_reshaped = c_sweep.view(n_points, 1, 1)
        p_o = self.p_open(c_reshaped, E_open, E_closed) # (n_points, 1)
        
        # Return physical concentrations and activities
        return c_sweep.cpu().numpy(), p_o.cpu().numpy()