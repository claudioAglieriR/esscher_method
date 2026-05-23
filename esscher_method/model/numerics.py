from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationConfig:
    # Max subdivisions for scipy adaptive quadrature in the Gil-Pelaez CDF integral.
    # 500 is sufficient for the bounded integration range (cdf_upper_bound = 100)
    # given the polynomial decay of the characteristic function.
    cdf_quad_limit: int = 500
    cdf_upper_bound: float = 100.0
    cdf_singularity_tol: float = 1e-14


@dataclass(frozen=True)
class DomainConfig:
    cgf_domain_eps: float = 1e-1


DEFAULT_INTEGRATION_CONFIG = IntegrationConfig()
DEFAULT_DOMAIN_CONFIG = DomainConfig()
