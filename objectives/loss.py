import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod

class BaseInformationLoss(nn.Module, ABC):
    """
    Abstract Base Class for Information Theoretic array optimization.
    Contains shared mathematical estimators for entropy.
    """
    def __init__(self):
        super().__init__()  

    def compute_knn_joint_entropy(self, activity: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Differentiable Kozachenko-Leonenko k-NN joint entropy estimator.
        Uses squared distances to prevent NaN gradients at zero.
        """
        B, N = activity.shape
        
        # 1. Pairwise squared distances: (B, B, N) -> (B, B)
        diff = activity.unsqueeze(1) - activity.unsqueeze(0)
        dists_sq = (diff ** 2).sum(dim=-1)
        
        # 2. k-th nearest neighbor
        k_dists_sq, _ = torch.topk(dists_sq, k + 1, dim=1, largest=False)
        rho_sq = k_dists_sq[:, k]
        rho_sq = torch.clamp(rho_sq, min=1e-12)
        
        # 3. Constants
        c_d = (math.pi ** (N / 2)) / math.gamma(N / 2 + 1)
        digamma_k = torch.digamma(torch.tensor(k, dtype=torch.float32, device=activity.device))
        digamma_B = torch.digamma(torch.tensor(B, dtype=torch.float32, device=activity.device))
        
        # 4. Entropy in nats, then bits
        log_rho_sum = 0.5 * torch.log(rho_sq).mean()
        h_nats = -digamma_k + digamma_B + math.log(c_d) + N * log_rho_sum
        return h_nats / math.log(2.0)

    def compute_kde_marginal_entropies(self, activity: torch.Tensor, bandwidth_factor: float = 1.06) -> torch.Tensor:
        """
        Differentiable 1D Gaussian KDE using the reflection method.
        Returns the individual entropy of each receptor: shape (N_Receptors,)
        """
        density = self._compute_kde(samples=activity,query_points=activity,bandwidth_factor=bandwidth_factor)
        
        log_prob = torch.log(density + 1e-8)
        return -torch.mean(log_prob, dim=0)
    
    def _compute_kde(self, samples: torch.Tensor, query_points: torch.Tensor,bandwidth_factor: float = 1.06) -> torch.Tensor:
        """
        Computes the Kernel Density Estimation of the samples, evaluated at query_points,
        using the Reflection Method to correct for [0, 1] boundary bias.
        """
        B_samples, R = samples.shape
        
        # 1. Compute Bandwidth (h)
        std = samples.std(dim=0) 
        h = bandwidth_factor * std * (B_samples ** (-0.2))
        h = torch.clamp(h, min=1e-4) 
        
        # 2. Broadcasting Setup
        X = query_points.unsqueeze(1)    # (B_query, 1, R)
        Y = samples.unsqueeze(0)         # (1, B_samples, R)
        h_reshaped = h.view(1, 1, R)
        norm_factor = math.sqrt(2 * math.pi)
        
        # 3. Calculate Distances (Original and Reflected)
        # the reflection is used to confined the kde with 0,1
        # Original distance
        diff_orig = X - Y
        # Reflected across 0 boundary: X - (-Y)
        diff_left = X + Y
        # Reflected across 1 boundary: X - (2 - Y)
        diff_right = X - (2.0 - Y)
        
        # 4. Apply Gaussian Kernels to all three
        u_orig = diff_orig / h_reshaped
        u_left = diff_left / h_reshaped
        u_right = diff_right / h_reshaped
        
        # Calculate exponentials separately
        kernel_orig = torch.exp(-0.5 * u_orig**2)
        kernel_left = torch.exp(-0.5 * u_left**2)
        kernel_right = torch.exp(-0.5 * u_right**2)
        
        # Sum and normalize OUT-OF-PLACE
        kernel_values = (kernel_orig + kernel_left + kernel_right) / norm_factor
        
        # 5. Sum over the samples to get the final density
        density = kernel_values.sum(dim=1) / (B_samples * h_reshaped.squeeze(0))
        
        return density
    
    @torch.no_grad()
    def make_stats(self, activity: torch.Tensor,bandwidth_factor: float = 1.06, k_knn:int = 5):
        # Detach just to be safe and ensure no graph is built
        # .detach() is nearly instantaneous
        act = activity.detach() 
        
        joint_h = self.compute_knn_joint_entropy(act, k_knn)
        marginals = self.compute_kde_marginal_entropies(act, bandwidth_factor)
        
        # Use .item() to pull the scalars out to Python
        return {
            "full_array_entropy":joint_h.item(), 
            "marginal_entropy":marginals.sum().item(), 
            "total_correlation":(marginals.sum() - joint_h).item()
        }
    
    @abstractmethod
    def forward(self, activity: torch.Tensor):
        pass


import torch
import torch.nn as nn

class BinaryProxyLoss(nn.Module):
    """
    Maximizes the exact analytical discrete Shannon entropy of the marginals
    while minimizing the linear covariance between different receptors.
    
    Expects `activity` to be probabilities (bounded between 0 and 1), 
    such as the output of a Sigmoid function.
    """
    def __init__(self, cov_weight: float = 1.0):
        super().__init__()
        self.cov_weight = cov_weight

    def _compute_analytical_marginal_entropies(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact discrete Shannon entropy for each receptor.
        activity shape: (Batch, R) - representing probability of firing.
        Returns shape: (R,) - entropy in bits.
        """
        # 1. The Marginal Probability of firing is just the mean over the batch
        # p_r shape: (R,)
        p_r = activity.mean(dim=0)
        
        # 2. Clamp to avoid log2(0)
        p_r = torch.clamp(p_r, min=1e-12, max=1.0 - 1e-12)
        
        # 3. Analytical Shannon Entropy: -p*log2(p) - (1-p)*log2(1-p)
        entropy = -p_r * torch.log2(p_r) - (1.0 - p_r) * torch.log2(1.0 - p_r)
        
        return entropy

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
        # 1. Compute Exact Marginal Entropies
        marginals = self._compute_analytical_marginal_entropies(activity)
        
        # We want to MAXIMIZE entropy, so we minimize the negative mean
        loss_entropy = -marginals.mean()
        
        # 2. Compute Covariance Penalty (we want to MINIMIZE this)
        loss_covariance = self._compute_covariance_penalty(activity)
        
        # 3. Total Loss
        total_loss = loss_entropy + (self.cov_weight * loss_covariance)
            
        return total_loss

    @torch.no_grad()
    def make_stats(self, activity: torch.Tensor):
        """
        Returns logging metrics for the training loop.
        """
        marginals = self._compute_analytical_marginal_entropies(activity)
        cov_penalty = self._compute_covariance_penalty(activity)
        
        return {
            "marginal_entropy_sum": marginals.sum().item(), 
            "marginal_entropy_mean": marginals.mean().item(),
            "covariance_penalty": cov_penalty.item()
        }

class DiscreteProxyLoss(nn.Module):
    """
    Maximizes the discrete Shannon entropy using a Differentiable Soft Histogram.
    Can handle an arbitrary number of bins (n_bins >= 2).
    """
    def __init__(self, cov_weight: float = 1.0, n_bins: int = 2, bin_temp: float = 0.05):
        """
        Args:
            cov_weight: Penalty weight for the covariance matrix.
            n_bins: Number of discrete activation levels (2 = binary).
            bin_temp: Temperature for the soft binning. Lower = sharper, harder bins.
        """
        super().__init__()
        self.cov_weight = cov_weight
        self.n_bins = n_bins
        self.bin_temp = bin_temp
        
        # Define the centers of the bins, evenly spaced between 0.0 and 1.0
        # Register as a buffer so it moves to the GPU automatically
        centers = torch.linspace(0.0, 1.0, n_bins)
        self.register_buffer('bin_centers', centers)

    def _compute_soft_histogram_entropy(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes the discrete Shannon entropy using soft assignments.
        activity shape: (Batch, R)
        Returns shape: (R,) - entropy in bits.
        """
        B, R = activity.shape
        
        # 1. Expand dimensions to compute distances to all bin centers
        # activity: (Batch, R, 1) | bin_centers: (1, 1, n_bins)
        act_expanded = activity.unsqueeze(-1)

        #centers = self.bin_centers.to(activity.device)
        centers_expanded = self.bin_centers.to(activity.device).unsqueeze(0).unsqueeze(0)
        
        # 2. Compute Squared Distance
        # Shape: (Batch, R, n_bins)
        dist_sq = (act_expanded - centers_expanded) ** 2
        
        # 3. Soft Assignment using Softmax over the bins dimension
        # A smaller bin_temp makes the assignment act more like a hard step-function
        soft_assign = torch.softmax(-dist_sq / self.bin_temp, dim=-1)
        
        # 4. Marginal Probability of falling into each bin
        # Average over the batch. Shape: (R, n_bins)
        p_marginal = soft_assign.mean(dim=0)
        
        # 5. Clamp to prevent log2(0) crashes
        p_marginal = torch.clamp(p_marginal, min=1e-12)
        
        # Calculate log base K using the change-of-base formula: ln(p) / ln(K)
        log_k_p = torch.log(p_marginal) / math.log(self.n_bins)
        
        # Exact Normalized Shannon Entropy
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
        mask = ~torch.eye(R, dtype=torch.bool, device=activity.device)
        
        return (cov_matrix[mask] ** 2).sum()

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
        B, R = activity.shape
        K = self.n_bins
        
        # ==========================================================
        # 1. Marginal Entropy (Exact)
        # ==========================================================
        marginals = self._compute_soft_histogram_entropy(activity)
        marginal_sum = marginals.sum().item()
        
        # ==========================================================
        # 2. Joint Entropy Calculation
        # ==========================================================
        # Re-calculate soft assignments to build the joint probabilities
        act_expanded = activity.unsqueeze(-1)
        centers = self.bin_centers.to(activity.device)
        dist_sq = (act_expanded - centers.unsqueeze(0).unsqueeze(0)) ** 2
        soft_assign = torch.softmax(-dist_sq / self.bin_temp, dim=-1) # Shape: (B, R, K)
        
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
            
            # Mask out strictly zero probabilities to prevent log2(0)
            p_a = p_a[p_a > 1e-12]            
            log_k_p = torch.log(p_a) / math.log(self.n_bins)
            joint_h = -torch.sum(p_a * log_k_p).item()
            
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
            log_k_p = torch.log(p_a + 1e-12) / math.log(self.n_bins)
            joint_h = -log_k_p.mean().item()

        # ==========================================================
        # 3. Return Dictionary
        # ==========================================================
        return {
            "full_array_entropy": joint_h, 
            "marginal_entropy": marginal_sum, 
            "total_correlation": marginal_sum - joint_h
        }

class ProxyInformationLoss(BaseInformationLoss):
    """
    Model A: Approximates Joint Entropy by maximizing independent marginal KDEs 
    and minimizing linear covariance. Faster, but ignores non-linear redundancy.
    """
    def __init__(self, cov_weight: float = 1.0, bandwidth_factor: float = 1.06):
        super().__init__()
        self.cov_weight = cov_weight
        self.bandwidth_factor=bandwidth_factor,

    def _compute_covariance_penalty(self, activity: torch.Tensor) -> torch.Tensor:
        B, R = activity.shape
        mean = activity.mean(dim=0, keepdim=True)
        centered = activity - mean
        cov_matrix = (centered.T @ centered) / (B - 1)
        mask = ~torch.eye(R, dtype=torch.bool, device=activity.device)
        return (cov_matrix[mask] ** 2).sum()

        
        #{
        #    "loss": total_loss.detach(),
        #    "proxy_entropy": loss_entropy.detach(),
        #    "covariance": loss_covariance.detach(),
        #    "true_joint_h": joint_h.detach(),
        #    "total_correlation": total_correlation.detach()
        #}

    def forward(self, activity: torch.Tensor):
        # 1. Compute Gradients via Proxy Method
        marginals = self.compute_kde_marginal_entropies(activity, self.bandwidth_factor)
        loss_entropy = -marginals.mean()
        loss_covariance = self._compute_covariance_penalty(activity)
        
        total_loss = loss_entropy + (self.cov_weight * loss_covariance)
    
            
        return total_loss

class ExactInformationLoss(BaseInformationLoss):
    """
    Model C: Directly maximizes the exact N-dimensional k-NN Joint Entropy. 
    Slower and slightly noisier gradients, but captures all non-linear redundancies.
    """
    def __init__(self, k_knn: int = 5):
        super().__init__()
        self.k_knn = k_knn
    def forward(self, activity: torch.Tensor):
        # 1. Compute Gradients directly through k-NN
        joint_h = self.compute_knn_joint_entropy(activity, self.k_knn)
        
        # Maximize entropy -> minimize negative
        total_loss = -joint_h
            
        return total_loss