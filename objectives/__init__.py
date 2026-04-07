# core/__init__.py

from .loss import (BaseInformationLoss,
                    ProxyInformationLoss,
                    ExactInformationLoss)

from .bin_loss import (compute_discrete_joint_entropy,
                        DiscreteProxyLoss,
                        DiscreteExactLoss)
from .tolerant_bin_loss import TolerantDiscreteProxyLoss
                   

__all__ = ["BaseInformationLoss",
            "ProxyInformationLoss",
            "ExactInformationLoss",
            "compute_discrete_joint_entropy",
            "DiscreteProxyLoss",
            "TolerantDiscreteProxyLoss",
            "DiscreteExactLoss"]