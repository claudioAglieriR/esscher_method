from __future__ import annotations

"""
Policy objects define:
  - default model parameters (initial guesses)
  - admissible bounds for calibration

The canonical default time step is DEFAULT_DELTA (in years).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

DEFAULT_TRADING_DAYS_PER_YEAR = 252.0
DEFAULT_DELTA = 1.0 / DEFAULT_TRADING_DAYS_PER_YEAR

__all__ = [
    "DEFAULT_TRADING_DAYS_PER_YEAR",
    "DEFAULT_DELTA",
    "MertonPolicy",
    "BilateralGammaPolicy",
    "VarianceGammaPolicy",
    "DEFAULT_MERTON_POLICY",
    "DEFAULT_BILATERAL_GAMMA_POLICY",
    "DEFAULT_VARIANCE_GAMMA_POLICY",
]


@dataclass(frozen=True)
class MertonPolicy:
    default_parameters: Dict[str, float] = field(default_factory=lambda: {"mu": 1.0, "sigma": 1e-2})

    mu_abs_bound: float = 4.0
    sigma_lower: float = 1e-8
    sigma_upper: float = 4.0

    intermediate_scale: float = 100.0
    sigma_floor: float = 1e-10

    def bounds(self) -> List[Tuple[float, float]]:
        return [
            (-self.mu_abs_bound, self.mu_abs_bound),
            (self.sigma_lower, self.sigma_upper),
        ]

    def intermediate_bounds(self, mu: float, sigma: float) -> List[Tuple[float, float]]:
        scale = self.intermediate_scale
        return [
            (-mu * scale, mu * scale),
            (self.sigma_floor, sigma * scale),
        ]


@dataclass(frozen=True)
class BilateralGammaPolicy:
    default_parameters: Dict[str, float] = field(
        default_factory=lambda: {
            "alpha_P": 10.0 + 1e-1,
            "lambda_P": 20.0 + 1.0 + 1e-1,
            "alpha_M": 10.0 + 1e-1,
            "lambda_M": 20.0 + 1e-1,
        }
    )

    lower_bound: float = 1e-2
    upper_bound: float = 1e3
    lambda_min_offset: float = 1.0

    def bounds(self) -> List[Tuple[float, float]]:
        lb = self.lower_bound
        ub = self.upper_bound
        return [
            (lb, ub),                        # alpha_P
            (self.lambda_min_offset + lb, ub),# lambda_P
            (lb, ub),                        # alpha_M
            (lb, ub),                        # lambda_M
        ]


@dataclass(frozen=True)
class VarianceGammaPolicy:
    default_parameters: Dict[str, float] = field(
        default_factory=lambda: {
            "alpha": 10.0 + 1e-1,
            "lambda_P": 20.0 + 1.0 + 1e-1,
            "lambda_M": 20.0 + 1e-1,
        }
    )

    lower_bound: float = 1e-2
    upper_bound: float = 1e3
    lambda_min_offset: float = 1.0

    def bounds(self) -> List[Tuple[float, float]]:
        lb = self.lower_bound
        ub = self.upper_bound
        return [
            (lb, ub),                        # alpha
            (self.lambda_min_offset + lb, ub),# lambda_P
            (lb, ub),                        # lambda_M
        ]


DEFAULT_MERTON_POLICY = MertonPolicy()
DEFAULT_BILATERAL_GAMMA_POLICY = BilateralGammaPolicy()
DEFAULT_VARIANCE_GAMMA_POLICY = VarianceGammaPolicy()
