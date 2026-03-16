# core/__init__.py

from .loss import (BaseInformationLoss,
                    ProxyInformationLoss,
                    ExactInformationLoss,
                    DiscreteProxyLoss)

__all__ = ["BaseInformationLoss",
            "ProxyInformationLoss",
            "ExactInformationLoss",
            "DiscreteProxyLoss"]