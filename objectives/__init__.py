# core/__init__.py

from .loss import (BaseInformationLoss,
                    ProxyInformationLoss,
                    ExactInformationLoss)

__all__ = ["BaseInformationLoss",
            "ProxyInformationLoss",
            "ExactInformationLoss"]