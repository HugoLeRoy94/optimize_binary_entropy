# core/__init__.py

from .loss import (BaseInformationLoss,
                    ProxyInformationLoss,
                    ExactInformationLoss)

from .bin_loss import (BinaryProxyLoss,
                        compute_discrete_joint_entropy,
                        DiscreteProxyLoss)                    

__all__ = ["BaseInformationLoss",
            "ProxyInformationLoss",
            "ExactInformationLoss",
            "BinaryProxyLoss",
            "compute_discrete_joint_entropy",
            "DiscreteProxyLoss"]