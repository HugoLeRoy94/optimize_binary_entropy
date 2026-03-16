import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseReceptor(nn.Module, ABC):
    """
    Abstract Base Class for Receptor Physics.
    Handles the common boilerplate of gathering units and computing dose-responses.
    """
    def __init__(self, n_units: int, k_sub: int = 5):
        super().__init__()
        self.n_units = n_units
        self.k_sub = k_sub

    @abstractmethod
    def p_open(self, c_reshaped: torch.Tensor, energies_k: torch.Tensor) -> torch.Tensor:
        """
        Calculates the activation probability.
        Must be implemented by the specific physics models.
        """
        pass

    @abstractmethod
    def _extract_mean_energies(self, env, receptor_indices, family_id, n_points) -> torch.Tensor:
        """
        Extracts and formats the mean interaction energies from the environment.
        """
        pass

    def forward(self, 
                energies: torch.Tensor, 
                concentrations: torch.Tensor, 
                receptor_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energies: Sampled interaction energies. 
                      Can be (Batch, n_units, 2) for MWC or (Batch, n_units) for Threshold.
            concentrations: (Batch,) - Sampled concentrations
            receptor_indices: (N_Receptors, k_sub) - Stoichiometry definitions
        """
        batch_size = energies.shape[0]
        n_receptors = receptor_indices.shape[0]
        
        # 1. Gather energies for specific receptors
        flat_indices = receptor_indices.view(-1)
        gathered_flat = energies[:, flat_indices]
        
        # 2. Reshape dynamically based on the input energy dimensions
        if energies.dim() == 3:
            # Shape: (Batch, R, k_sub, 2)
            energies_k = gathered_flat.view(batch_size, n_receptors, self.k_sub, energies.shape[-1])
        else:
            # Shape: (Batch, R, k_sub)
            energies_k = gathered_flat.view(batch_size, n_receptors, self.k_sub)
        
        # 3. Compute Activity
        c_reshaped = concentrations.view(batch_size, 1, 1)
        p_o = self.p_open(c_reshaped, energies_k)        
        
        return torch.clamp(p_o, 0.0, 1.0)

    @torch.no_grad()
    def get_dose_response(self, env, receptor_indices, family_id, n_points=200, method='absolute'):
        c_sweep, _ = env.get_concentration_sweep(family_id, n_points)
        c_reshaped = c_sweep.view(n_points, 1, 1)

        # Delegate the energy extraction to the subclass
        gathered_energies = self._extract_mean_energies(env, receptor_indices, family_id, n_points)
        
        p_o = self.p_open(c_reshaped, gathered_energies) 
        
        if method == "self_normalized":
            p_max = torch.max(p_o, dim=0, keepdim=True).values
            p_o =  p_o / (p_max + 1e-8)
        
        return c_sweep.cpu().numpy(), p_o.cpu().numpy()


class MWCReceptor(BaseReceptor):
    """
    The classic MWC Model.
    Expects energies of shape (Batch, n_units, 2) where index 0 is open, 1 is closed.
    """
    def p_open(self, c_reshaped: torch.Tensor, energies_k: torch.Tensor):
        E_open = energies_k[..., 0]
        E_closed = energies_k[..., 1]
        
        ln_c = torch.log(c_reshaped + 1e-12)
        
        log_w_open_per_unit = ln_c - E_open
        ln_W_open = torch.sum(log_w_open_per_unit, dim=-1)
        
        log_w_closed_per_unit = torch.nn.functional.softplus(ln_c - E_closed)
        ln_W_closed = torch.sum(log_w_closed_per_unit, dim=-1)
        
        return torch.sigmoid(ln_W_open - ln_W_closed)

    def _extract_mean_energies(self, env, receptor_indices, family_id, n_points):
        N_Receptors = receptor_indices.shape[0]
        mu_family = env.interaction_mu[:, family_id, :] 
        mu_receptor = mu_family[receptor_indices]
        return mu_receptor.unsqueeze(0).expand(n_points, N_Receptors, self.k_sub, 2)


class BinaryReceptor(BaseReceptor):
    """
    Simplified Threshold (Binary) Model.
    Expects energies of shape (Batch, n_units) representing the open-state affinity.
    """
    def __init__(self, n_units: int, k_sub: int = 5, temperature: float = 0.1):
        super().__init__(n_units, k_sub)
        self.temperature = temperature

    def p_open(self, c_reshaped: torch.Tensor, energies_k: torch.Tensor):
        # ln(EC50) is the simple average of the subunit open energies
        log_ec50 = energies_k.mean(dim=-1) # Shape: (Batch, R)
        
        # Match dimensions: (Batch, 1, 1) -> (Batch, 1) to broadcast smoothly over R
        ln_c = torch.log(c_reshaped + 1e-12).squeeze(-1) 
        
        # Temperature-Scaled Binary Activation
        return torch.sigmoid((ln_c - log_ec50) / self.temperature)

    def _extract_mean_energies(self, env, receptor_indices, family_id, n_points):
        N_Receptors = receptor_indices.shape[0]
        
        # Fetch directly from the 2D energy matrix
        mu_family = env.interaction_mu[:, family_id]
        
        # Fallback if the environment hasn't been updated yet and still returns 3D
        if mu_family.dim() > 1 and mu_family.shape[-1] == 2:
            mu_family = mu_family[..., 0] 
            
        mu_receptor = mu_family[receptor_indices]
        return mu_receptor.unsqueeze(0).expand(n_points, N_Receptors, self.k_sub)