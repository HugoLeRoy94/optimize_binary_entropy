import torch
import torch.nn as nn
import math

def compute_discrete_joint_entropy(soft_assign: torch.Tensor) -> torch.Tensor:
    """
    Computes the joint entropy of a discrete system from soft assignments.
    Switches between exact enumeration for small state spaces and Monte Carlo
    estimation for large state spaces.
    """
    B, R, K = soft_assign.shape
    
    # Dynamic switch based on computational complexity
    if K ** R <= 1024:
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
        
        # 3. Fast Matrix Multiplication to compute log P(state_j | ligand_i)
        S_one_hot = torch.nn.functional.one_hot(sampled_states, num_classes=K).float()
        log_probs_flat = log_probs.view(B, -1)
        S_one_hot_flat = S_one_hot.view(B, -1)
        
        # Subsample states to prevent OOM on massive batches
        M = min(B, 2048)
        if M < B:
            indices = torch.randperm(B, device=soft_assign.device)[:M]
            S_one_hot_flat = S_one_hot_flat[indices]
            
        log_p_a_given_x = torch.matmul(log_probs_flat, S_one_hot_flat.T) # (B_x, M)
        
        # 4. Average across the batch of ligands to get the true P(state_j)
        p_a_given_x = torch.exp(log_p_a_given_x) # (B_x, B_a)
        p_a = p_a_given_x.mean(dim=0) # (M,)
        
        # 5. Monte Carlo average: Expected value of [-log_K P(state)]
        p_a_safe = torch.clamp(p_a, min=1e-12)
        log_k_p = torch.log(p_a_safe) / math.log(K)
        joint_h = -log_k_p.mean()

    return joint_h

class DiscreteExactLoss(nn.Module):
    """
    Maximizes the exact discrete joint entropy of the array.
    Ideal for systems where components must be correlated (like a thermometer code).
    """
    def __init__(self, n_bins: int = 2, bin_temp: float = 0.05):
        super().__init__()
        self.n_bins = n_bins
        self.bin_temp = bin_temp
        centers = torch.linspace(0.0, 1.0, n_bins)
        self.register_buffer('bin_centers', centers)

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        if self.n_bins == 2:
            # For binary systems, activity is exactly P(fire). This avoids vanishing gradients.
            return torch.stack([1.0 - activity, activity], dim=-1)
        act_expanded = activity.unsqueeze(-1)
        centers_expanded = self.bin_centers.view(1, 1, -1)
        dist_sq = (act_expanded - centers_expanded) ** 2
        return torch.softmax(-dist_sq / self.bin_temp, dim=-1)

    def forward(self, activity: torch.Tensor):
        soft_assign = self.compute_soft_assignment(activity)
        joint_h = compute_discrete_joint_entropy(soft_assign)
        return -joint_h  # Maximize joint entropy

class DiscreteProxyLoss(nn.Module):
    """
    Maximizes the discrete Shannon entropy of the marginals using a Differentiable Soft Histogram,
    while minimizing a penalty term to encourage receptor independence or diversity.
    """
    def __init__(self, cov_weight: float = 1.0, n_bins: int = 2, bin_temp: float = 0.05, penalty_type: str = 'repulsion'):
        """
        Args:
            cov_weight: Weight for the penalty term.
            n_bins: Number of discrete activation levels (2 = binary).
            bin_temp: Temperature for the soft binning. Lower = sharper, harder bins.
            penalty_type: The type of penalty to apply. Can be 'repulsion' (penalizes
                          similar activation profiles) or 'covariance' (penalizes linear
                          correlation). Defaults to 'repulsion'.
        """
        super().__init__()
        self.cov_weight = cov_weight
        self.n_bins = n_bins
        self.bin_temp = bin_temp
        self.penalty_type = penalty_type

        if self.penalty_type not in ['repulsion', 'covariance']:
            raise ValueError("penalty_type must be 'repulsion' or 'covariance'")
        
        # Define the centers of the bins, evenly spaced between 0.0 and 1.0
        # Register as a buffer so it moves to the GPU automatically
        centers = torch.linspace(0.0, 1.0, n_bins)
        self.register_buffer('bin_centers', centers)

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

    def _compute_repulsion_penalty(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Penalizes receptors for having identical continuous activation profiles.
        A perfectly shifted thermometer code has low overlap (low penalty),
        while identical receptors have max overlap (high penalty).
        """
        B, R = activity.shape
        A = activity.T # (R, B)
        
        # Pairwise squared Euclidean distance: ||A_i - A_j||^2
        A_sq = (A ** 2).sum(dim=1, keepdim=True) # (R, 1)
        dist_matrix = (A_sq + A_sq.T - 2.0 * torch.matmul(A, A.T)) / B
        
        # Repulsion kernel: exp(-dist / tau)
        tau = 0.05 
        repulsion = torch.exp(-dist_matrix / tau)
        
        mask = ~torch.eye(R, dtype=torch.bool, device=activity.device)
        
        return repulsion[mask].sum()

    def _compute_covariance_penalty(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Minimizes the off-diagonal terms of the covariance matrix 
        to encourage independent receptors.
        """
        B, R = activity.shape
        
        # Center the probabilities
        mean = activity.mean(dim=0, keepdim=True)
        centered = activity - mean
        
        # Compute Covariance Matrix: (R, B) @ (B, R) -> (R, R)
        cov_matrix = (centered.T @ centered) / (B - 1)
        
        # Create a mask to ignore the diagonal (variance)
        mask = ~torch.eye(R, dtype=torch.bool, device=activity.device)
        
        # Sum of squared off-diagonal elements
        return (cov_matrix[mask] ** 2).sum()

    def forward(self, activity: torch.Tensor):
        # 1. Compute Entropy using Differentiable Bins
        marginals = self._compute_soft_histogram_entropy(activity)
        loss_entropy = -marginals.mean() # Maximize entropy
        
        # 2. Compute selected penalty
        if self.penalty_type == 'repulsion':
            penalty = self._compute_repulsion_penalty(activity)
        else: # covariance
            penalty = self._compute_covariance_penalty(activity)
        
        # 3. Total Loss
        total_loss = loss_entropy + (self.cov_weight * penalty)
            
        return total_loss
