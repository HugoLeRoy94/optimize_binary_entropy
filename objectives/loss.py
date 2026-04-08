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
    
    @abstractmethod
    def forward(self, activity: torch.Tensor):
        pass

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