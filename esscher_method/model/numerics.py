from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationConfig:
    cdf_quad_limit: int = 1_000_000_000 # TODO : reset to 500
    cdf_upper_bound: float = 100.0
    cdf_singularity_tol: float = 1e-14
    whittaker_cdf_lower: float = -100.0


@dataclass(frozen=True)
class DomainConfig:
    cgf_domain_eps: float = 1e-1


DEFAULT_INTEGRATION_CONFIG = IntegrationConfig()
DEFAULT_DOMAIN_CONFIG = DomainConfig()
