import math
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple
from abc import ABC, abstractmethod

class UniformNBall:
    """
    Custom differentiable sampler for an N-dimensional solid sphere (N-ball).
    """
    def __init__(self, loc: torch.Tensor, radius: float, dim: int):
        self.loc = loc          # Shape: (Batch, latent_dim)
        self.radius = radius    # Scalar
        self.dim = dim          # Scalar (latent_dim)

    def rsample(self) -> torch.Tensor:
        # 1. Sample a perfectly random direction
        direction = torch.randn_like(self.loc)
        direction = torch.nn.functional.normalize(direction, p=2, dim=-1)
        
        # 2. Sample the radius to ensure uniform volume density
        # Draw U ~ Uniform(0, 1)
        u = torch.rand(self.loc.shape[0], 1, device=self.loc.device)
        r = self.radius * (u ** (1.0 / self.dim))
        
        # 3. Scale and shift
        return self.loc + (direction * r)

class ConcentrationModel(nn.Module, ABC):
    """
    Abstract Base Class for different concentration strategies.
    Subclass this to create LogNormal, Normal, Bimodal, etc.
    """
    @abstractmethod
    def sample(self, batch_size: int, family_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns concentrations for the given family_ids.
        Shape: (batch_size,)
        """
        pass        
    @abstractmethod
    def get_expected_log_c(self) -> torch.Tensor:
        """
        Returns the expected natural logarithm of the concentration for each family.
        Shape: (n_families,)
        """
        pass
    @abstractmethod
    def get_distribution(self, family_id: int) -> dist.Distribution:
        """Returns the torch distribution object for a specific family."""
        pass
    @abstractmethod
    def get_sweep_and_pdf(self, family_id: int, n_points: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the physical concentration sweep and its PDF."""
        pass

class LogNormalConcentration(ConcentrationModel):
    """
    Classic Biophysics assumption: c spans orders of magnitude.
    log10(c) ~ Normal(mu, sigma)
    """
    def __init__(self, n_families: int, init_mean=-6.0, init_scale=1.0):
        super().__init__()
        # Initialize around 10^-6 M (1 microM)
        #self.mu = nn.Parameter(torch.ones(n_families) * init_mean)
        #self.log_sigma = nn.Parameter(torch.ones(n_families) * math.log(init_scale))
        
        mu_tensor = torch.tensor(init_mean, dtype=torch.float32)
        if mu_tensor.ndim == 0:
            mu_tensor = torch.ones(n_families) * mu_tensor
            
        sigma_tensor = torch.tensor(init_scale, dtype=torch.float32)
        
        self.register_buffer('mu', mu_tensor)
        self.register_buffer('log_sigma', torch.ones(n_families) * torch.log(sigma_tensor))

    def sample(self, batch_size, family_ids):
        # Gather params for this batch
        batch_mu = self.mu[family_ids]
        batch_sigma = torch.exp(self.log_sigma[family_ids])
        
        # Sample Log-Space
        dist_log = dist.Normal(batch_mu, batch_sigma)
        log_c = dist_log.rsample()
        
        # Convert to Real-Space
        return torch.pow(10.0, log_c)
        
    def get_expected_log_c(self):
        # mu is log10(c). To get natural log ln(c): ln(c) = log10(c) * ln(10)
        return self.mu * math.log(10.0)
    
    def get_entropy_linear(self):
        """
        Entropy of the base-10 Log-Normal distribution.

        """
        # 1. Entropy of the underlying Normal distribution (log10 space) in bits
        sigma = torch.exp(self.log_sigma)
        h_normal = torch.log2(sigma * math.sqrt(2 * math.pi * math.e))
        
        # 2. Add the Jacobian contribution for the 10^x transformation
        # E[ln(c)] = mu * ln(10)
        jacobian_term = (self.mu * math.log(10.0) + math.log(math.log(10.0))) / math.log(2.0)
        
        return h_normal + jacobian_term
    
    def get_entropy_log(self):
        """
        Entropy of a Normal distribution in bits.
        """
        sigma = torch.exp(self.log_sigma)
        return torch.log2(sigma * math.sqrt(2 * math.pi * math.e))

    @torch.no_grad()
    def get_distribution(self, family_id: int):
        mu = self.mu[family_id]
        sigma = torch.exp(self.log_sigma[family_id])
        return dist.Normal(mu, sigma) # Distribution of log10(c)
        
    @torch.no_grad()
    def get_sweep_and_pdf(self, family_id: int, n_points: int = 200):
        d = self.get_distribution(family_id)
        # Sweep in log10 space
        x_log10 = torch.linspace(d.mean - 3*d.stddev, d.mean + 3*d.stddev, n_points, device=self.mu.device)
        pdf = torch.exp(d.log_prob(x_log10))
        # Convert to physical concentration (Molar)
        c_sweep = 10.0 ** x_log10
        return c_sweep, pdf

class NormalConcentration(ConcentrationModel):
    """
    Simple Gaussian assumption.
    c ~ Normal(mu, sigma) clamped at 0.
    """
    def __init__(self, n_families: int, init_mean=10**-6, init_scale=10**-7):
        super().__init__()
        #self.mu = nn.Parameter(torch.ones(n_families) * init_mean)
        #self.log_sigma = nn.Parameter(torch.ones(n_families) * math.log(init_scale))
        self.register_buffer('mu', torch.ones(n_families) * init_mean)
        self.register_buffer('log_sigma', torch.ones(n_families) * math.log(init_scale))

    def sample(self, batch_size, family_ids):
        batch_mu = self.mu[family_ids]
        batch_sigma = torch.exp(self.log_sigma[family_ids])
        
        c = dist.Normal(batch_mu, batch_sigma).rsample()
        return torch.clamp(c, min=1e-12) # Physics constraint (clamp > 0 for safe log)
        
    def get_expected_log_c(self):
        # Approximate expected log(c) as log(mu)
        return torch.log(torch.clamp(self.mu, min=1e-12))
    
    def get_entropy(self):
        """
        Entropy of a Normal distribution in bits.
        """
        sigma = torch.exp(self.log_sigma)
        return torch.log2(sigma * math.sqrt(2 * math.pi * math.e))

    @torch.no_grad()
    def get_distribution(self, family_id: int):
        mu = self.mu[family_id]
        sigma = torch.exp(self.log_sigma[family_id])
        return dist.Normal(mu, sigma) # Distribution of c
    
    @torch.no_grad()
    def get_sweep_and_pdf(self, family_id: int, n_points: int = 200):
        d = self.get_distribution(family_id)
        # Sweep already in linear physical space
        c_sweep = torch.linspace(d.mean - 3*d.stddev, d.mean + 3*d.stddev, n_points, device=self.mu.device)
        pdf = torch.exp(d.log_prob(c_sweep))
        return c_sweep, pdf

class LigandEnvironment(nn.Module):
    def __init__(self, n_units: int, n_families: int, conc_model: ConcentrationModel, 
                 latent_dim: int = 3, shape_sigma: float = 0.5, distribution_type: str = 'gaussian'):
        """
        Args:
            n_units: Number of protein units
            n_families: Number of ligand families
            conc_model: An INSTANCE of a ConcentrationModel subclass
            latent_dim: Dimensionality of the chemical latent space (trade-off space)
            shape_sigma: The fixed spatial spread (fuzziness) of the ligand families
            distribution_type: 'gaussian' or 'uniform'
        """
        super().__init__()
        self.n_units = n_units
        self.n_families = n_families
        self.latent_dim = latent_dim
        self.shape_sigma = shape_sigma
        
        if distribution_type not in ['gaussian', 'uniform']:
            raise ValueError("distribution_type must be 'gaussian' or 'uniform'")
        self.distribution_type = distribution_type
        
        # 1. Inject the Concentration Strategy
        self.concentration_model = conc_model
        
        # ----------------------------------------------------------------------
        # MECHANISTIC LATENT SPACE INITIALIZATION
        # ----------------------------------------------------------------------
        
        # 1. The Octopus Adapts (Learnable Unit Coordinates)
        self.unit_latent = nn.Parameter(torch.randn(n_units, latent_dim))
        
        # 2. The Environment is Fixed (Family Prototype Coordinates)
        fixed_families = torch.randn(n_families, latent_dim)
        fixed_families = torch.nn.functional.normalize(fixed_families, p=2, dim=1)
        self.register_buffer('family_latent', fixed_families)
        
        # 3. Unit-specific Base Energies (Maximum intrinsic affinity)
        global_avg_log_c = self.concentration_model.get_expected_log_c().mean().item()
        self.base_energy_u = nn.Parameter(torch.ones(n_units) * global_avg_log_c)

    @property
    def interaction_mu(self) -> torch.Tensor:
        """
        Compatibility property for plotting dose-response curves.
        Returns the MEAN open-state energy for each unit against each family's exact center.
        Shape: (n_units, n_families)
        """
        diff = self.unit_latent.unsqueeze(1) - self.family_latent.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1) 
        
        mu_open = self.base_energy_u.unsqueeze(1) + dist_sq
        return mu_open

    def sample_batch(self, batch_size: int):
        device = self.unit_latent.device
        family_ids = torch.randint(0, self.n_families, (batch_size,), device=device)
        return self._sample_from_ids(batch_size, family_ids)

    def sample_specific_family(self, batch_size: int, family_id: int):
        device = self.unit_latent.device
        f_ids = torch.full((batch_size,), family_id, dtype=torch.long, device=device)
        return self._sample_from_ids(batch_size, f_ids)

    def _sample_from_ids(self, batch_size: int, family_ids: torch.Tensor):
        # 1. Sample physical concentrations
        concs = self.concentration_model.sample(batch_size, family_ids)
        
        # 2. Get the prototype centers for this batch: (Batch, latent_dim)
        batch_centers = self.family_latent[family_ids] 
        
        # 3. Draw Ligand Coordinates directly from a PyTorch Distribution
        if self.distribution_type == 'gaussian':
            # Isotropic Gaussian: variance is shape_sigma^2 in all directions
            ligand_dist = dist.Normal(loc=batch_centers, scale=self.shape_sigma)
            v_ligands = ligand_dist.rsample()
            
        elif self.distribution_type == 'uniform_cube':
            # Uniform hypercube centered at batch_centers with side length 2 * shape_sigma
            low = batch_centers - self.shape_sigma
            high = batch_centers + self.shape_sigma
            ligand_dist = dist.Uniform(low=low, high=high)
            v_ligands = ligand_dist.rsample()
        elif self.distribution_type == 'uniform':
            # Use our custom uniform N-ball sampler!
            ligand_dist = UniformNBall(loc=batch_centers, radius=self.shape_sigma, dim=self.latent_dim)
            v_ligands = ligand_dist.rsample()
            
        # 4. Calculate Energies based on Distance
        diff = v_ligands.unsqueeze(1) - self.unit_latent.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1) # Shape: (Batch, n_units)
        
        E_open = self.base_energy_u + dist_sq
        
        return E_open, concs, family_ids

    @torch.no_grad()
    def get_concentration_sweep(self, family_id: int, n_points: int = 200):
        return self.concentration_model.get_sweep_and_pdf(family_id, n_points)