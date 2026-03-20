import torch
import torch.nn as nn
import math
import sys
sys.path.append('../src')
from src.environment import LigandEnvironment, LogNormalConcentration, NormalConcentration

def build_overlap_matrix(receptor_indices: torch.Tensor, n_units: int) -> torch.Tensor:
    """
    Computes the fractional subunit overlap between every pair of receptors.
    receptor_indices shape: (N_sensors, k_sub)
    Returns shape: (N_sensors, N_sensors)
    """
    N_sensors, k_sub = receptor_indices.shape
    
    # 1. Count occurrences of each unit for each receptor
    # Shape: (N_sensors, n_units)
    counts = torch.zeros(N_sensors, n_units, device=receptor_indices.device)
    for i in range(N_sensors):
        counts[i] = torch.bincount(receptor_indices[i], minlength=n_units)
        
    # 2. Compute multiset intersection using broadcasting
    counts_i = counts.unsqueeze(1) # (N_sensors, 1, n_units)
    counts_j = counts.unsqueeze(0) # (1, N_sensors, n_units)
    
    # The number of shared subunits is the sum of the minimum counts
    shared_subunits = torch.min(counts_i, counts_j).sum(dim=-1) # (N_sensors, N_sensors)
    
    # 3. Normalize by k_sub to get a fraction from 0.0 to 1.0
    overlap_matrix = shared_subunits.float() / k_sub
    
    return overlap_matrix

def compute_discrete_joint_entropy(soft_assign: torch.Tensor) -> torch.Tensor:
    """
    Computes the joint entropy of a discrete system from soft assignments.
    Switches between exact enumeration for small state spaces and Monte Carlo
    estimation for large state spaces.
    """
    B, R, K = soft_assign.shape
    
    # Dynamic switch based on computational complexity
    if K ** R <= 1_000_000:
        # ------------------------------------------------------
        # METHOD A: Exact Enumeration (For small arrays)
        # ------------------------------------------------------
        # We iteratively build the exact (B, K^R) probability tensor
        joint_p = soft_assign[:, 0, :] # Start with receptor 0: (B, K)
        
        for r in range(1, R):
            # Multiply current combinations by the probabilities of the next receptor
            # (B, K^{r}, 1) * (B, 1, K) -> flat to (B, K^{r+1})
            joint_p = (joint_p.unsqueeze(-1) * soft_assign[:, r, :].unsqueeze(1)).view(B, -1)
        
        # Average across the batch to get the true probability of every possible state
        p_a = joint_p.mean(dim=0) # Shape: (K^R,)
        
        # Use clamp to prevent log2(0) while maintaining stable gradients
        p_a_safe = torch.clamp(p_a, min=1e-12)
        log_k_p = torch.log(p_a_safe) / math.log(K)
        joint_h = -torch.sum(p_a * log_k_p)
        
    else:
        # ------------------------------------------------------
        # METHOD B: Monte Carlo Estimation (For large arrays)
        # ------------------------------------------------------
        # 1. Sample 1 actual discrete array state per ligand in the batch
        dist = torch.distributions.Categorical(probs=soft_assign)
        sampled_states = dist.sample() # Shape: (B_a, R)
        
        # 2. Extract the log probabilities: P(A_r | ligand_i)
        log_probs = torch.log(soft_assign + 1e-12) # Shape: (B_x, R, K)
        
        # 3. Vectorized Gathering (Calculate P(sampled_state | ligand) for all ligands)
        # Expand dimensions so we can evaluate all B_a states against all B_x ligands
        log_probs_exp = log_probs.unsqueeze(1).expand(B, B, R, K)
        states_exp = sampled_states.unsqueeze(0).unsqueeze(-1).expand(B, B, R, 1)
        
        # Gather the exact log probs for the sampled states
        gathered_log_probs = torch.gather(log_probs_exp, 3, states_exp).squeeze(-1) # (B_x, B_a, R)
        
        # Sum over receptors to get log P(state_j | ligand_i)
        log_p_a_given_x = gathered_log_probs.sum(dim=-1) # (B_x, B_a)
        
        # 4. Average across the batch of ligands to get the true P(state_j)
        p_a_given_x = torch.exp(log_p_a_given_x) # (B_x, B_a)
        p_a = p_a_given_x.mean(dim=0) # (B_a,)
        
        # 5. Monte Carlo average: Expected value of [-log_K P(state)]
        p_a_safe = torch.clamp(p_a, min=1e-12)
        log_k_p = torch.log(p_a_safe) / math.log(K)
        joint_h = -log_k_p.mean()

    return joint_h

class TolerantDiscreteProxyLoss(nn.Module):
    """
    Maximizes the discrete Shannon entropy using a Differentiable Soft Histogram.
    Can handle an arbitrary number of bins (n_bins >= 2).
    """
    def __init__(self, 
                env : LigandEnvironment,
                receptor_indices: torch.tensor,
                n_units:int,
                cov_weight: float = 1.0, 
                n_bins: int = 2, 
                bin_temp: float = 0.05):
        """
        Args:
            cov_weight: Penalty weight for the covariance matrix.
            n_bins: Number of discrete activation levels (2 = binary).
            bin_temp: Temperature for the soft binning. Lower = sharper, harder bins.
        """
        super().__init__()
        self.k_sub = receptor_indices.shape[1]
        self.env = env
        self.cov_weight = cov_weight
        self.n_bins = n_bins
        self.bin_temp = bin_temp
        
        # Define the centers of the bins, evenly spaced between 0.0 and 1.0
        # Register as a buffer so it moves to the GPU automatically
        centers = torch.linspace(0.0, 1.0, n_bins)
        self.register_buffer('bin_centers', centers)

        self.register_buffer('overlap_matrix', build_overlap_matrix(receptor_indices,n_units))

    def _get_dynamic_tolerance(self) -> torch.Tensor:
        # ---------------------------------------------------------------------
        # 1. Estimate Concentration Variance (Var(ln c))
        # ---------------------------------------------------------------------
        conc_model = self.env.concentration_model
        
        if isinstance(conc_model, LogNormalConcentration):
            sigma_log10 = torch.exp(conc_model.log_sigma).mean()
            var_c = (sigma_log10 * math.log(10.0)) ** 2
            
        elif isinstance(conc_model, NormalConcentration):
            mu = conc_model.mu.mean()
            sigma = torch.exp(conc_model.log_sigma).mean()
            var_c = (sigma / torch.clamp(mu, min=1e-12)) ** 2
            
        else:
            var_c = torch.tensor(0.0, device=self.overlap_matrix.device)

        # ---------------------------------------------------------------------
        # 2. Estimate Energy Variance (Var(E_o))
        # ---------------------------------------------------------------------
        latent_dim = self.env.latent_dim
        shape_sigma = self.env.shape_sigma
        
        var_o = 2.0 * latent_dim * (shape_sigma ** 4)

        # ---------------------------------------------------------------------
        # 3. Adjusted Price's Theorem
        # ---------------------------------------------------------------------
        var_o_k = var_o / self.k_sub
        
        # A. Calculate theoretical covariance for the actual overlap
        rho_m = (var_c + self.overlap_matrix * var_o_k) / (var_c + var_o_k)
        rho_m = torch.clamp(rho_m, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        theoretical_cov = (1.0 / (2.0 * math.pi)) * torch.asin(rho_m)
        
        # B. Calculate baseline covariance for ZERO overlap (m=0)
        rho_baseline = var_c / (var_c + var_o_k)
        
        # If var_c was somehow 0.0 (a float instead of a tensor), ensure it's a tensor safely:
        if not isinstance(rho_baseline, torch.Tensor):
            rho_baseline = torch.tensor(rho_baseline, device=self.overlap_matrix.device)
            
        rho_baseline = torch.clamp(rho_baseline, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        baseline_cov = (1.0 / (2.0 * math.pi)) * torch.asin(rho_baseline)
        
        # C. Subtract baseline to guarantee T=0 when overlap=0
        adjusted_tolerance = torch.relu(theoretical_cov - baseline_cov)
        
        return adjusted_tolerance

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes soft assignments of continuous activity to discrete bins.

        Args:
            activity (torch.Tensor): Continuous activity tensor of shape (Batch, R).

        Returns:
            torch.Tensor: Soft assignment tensor of shape (Batch, R, n_bins).
        """
        if self.n_bins == 2:
            return torch.stack([1.0 - activity, activity], dim=-1)
        act_expanded = activity.unsqueeze(-1)
        centers_expanded = self.bin_centers.view(1, 1, -1)
        
        dist_sq = (act_expanded - centers_expanded) ** 2
        soft_assign = torch.softmax(-dist_sq / self.bin_temp, dim=-1)
        
        return soft_assign

    def compute_soft_marginal_probabilities(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes marginal probabilities for each bin using a soft-assignment.

        Args:
            activity (torch.Tensor): Continuous activity tensor of shape (Batch, R).

        Returns:
            torch.Tensor: Marginal probabilities of shape (R, n_bins).
        """
        soft_assign = self.compute_soft_assignment(activity)
        p_marginal = soft_assign.mean(dim=0)
        return p_marginal

    def _compute_soft_histogram_entropy(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes the discrete Shannon entropy using soft assignments.
        activity shape: (Batch, R)
        Returns shape: (R,) - entropy in bits.
        """
        # 1. Get marginal probabilities from the shared function
        p_marginal = self.compute_soft_marginal_probabilities(activity)
        
        # 2. Clamp to prevent log2(0) crashes
        p_marginal = torch.clamp(p_marginal, min=1e-12)
        
        # 3. Calculate log base K using the change-of-base formula: ln(p) / ln(K)
        log_k_p = torch.log(p_marginal) / math.log(self.n_bins)
        
        # 4. Exact Normalized Shannon Entropy
        entropy = -torch.sum(p_marginal * log_k_p, dim=-1)
        
        return entropy

    def _compute_covariance_penalty(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Minimizes the off-diagonal terms of the linear covariance matrix.
        We apply this to the raw continuous activity rather than the bins 
        because it is vastly more computationally efficient and serves as a perfect proxy.
        """
        B, R = activity.shape
        mean = activity.mean(dim=0, keepdim=True)
        centered = activity - mean
        
        cov_matrix = (centered.T @ centered) / (B - 1)
        # Fetch the dynamically computed tolerance
        dynamic_tolerance = self._get_dynamic_tolerance()
        
        # Apply the tolerance mask
        excess_cov = torch.relu(torch.abs(cov_matrix) - dynamic_tolerance)
        
        # Zero out the diagonal (we do not penalize variance)
        mask = ~torch.eye(R, dtype=torch.bool, device=activity.device)
        
        return (excess_cov[mask] ** 2).sum()

    def forward(self, activity: torch.Tensor):
        # 1. Compute Entropy using Differentiable Bins
        marginals = self._compute_soft_histogram_entropy(activity)
        loss_entropy = -marginals.mean() # Maximize entropy
        
        # 2. Compute Covariance Penalty
        loss_covariance = self._compute_covariance_penalty(activity)
        
        # 3. Total Loss
        total_loss = loss_entropy + (self.cov_weight * loss_covariance)
            
        return total_loss

    @torch.no_grad()
    def make_stats(self, activity: torch.Tensor):
        # ==========================================================
        # 1. Marginal Entropy (Exact)
        # ==========================================================
        marginals = self._compute_soft_histogram_entropy(activity)
        marginal_sum = marginals.sum().item()
        
        # ==========================================================
        # 2. Joint Entropy Calculation
        # ==========================================================
        # Re-calculate soft assignments to build the joint probabilities
        soft_assign = self.compute_soft_assignment(activity) # Shape: (B, R, K)
        
        joint_h = compute_discrete_joint_entropy(soft_assign)
        
        # ==========================================================
        # 3. Return Dictionary
        # ==========================================================
        return {
            "full_array_entropy": joint_h.item() if isinstance(joint_h, torch.Tensor) else joint_h, 
            "marginal_entropy": marginal_sum, 
            "total_correlation": marginal_sum - (joint_h.item() if isinstance(joint_h, torch.Tensor) else joint_h)
        }
