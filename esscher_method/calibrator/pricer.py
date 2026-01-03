# esscher_method/calibrator/pricer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.integrate import quad

from esscher_method.model.model import Model


Number = Union[int, float, np.number]


@dataclass(frozen=True)
class LewisPricerConfig:
    """Numerical settings for the Lewis Fourier integral."""
    integration_upper_bound: float = float(2**10)
    quad_limit: int = int(20_000) #TODO : reset to 500


class LewisEuropeanTargetPricer:
    """European call pricer using the Lewis Fourier representation under risk-neutral dynamics."""

    def __init__(
        self,
        model: Model,
        *,
        T: Optional[float] = None,
        K: Optional[Number] = None,
        risk_free_rate: Optional[float] = None,
        target_price: Optional[Number] = None,
        config: Optional[LewisPricerConfig] = None,
    ) -> None:
        if model is None:
            raise ValueError("model must not be None.")

        self.model = model
        self.T = T
        self.K = K
        self.target_price = target_price
        self.config = config or LewisPricerConfig()
        self.risk_free_rate = 0.0 if risk_free_rate is None else float(risk_free_rate)

    @staticmethod
    def _to_scalar(value: Number) -> float:
        """Convert a scalar-like value (python number or numpy scalar/array) to a float."""
        return float(np.asarray(value, dtype=float).reshape(-1)[0])

    def _validate_inputs(self) -> None:
        """Validate that required pricing inputs are available and well-formed."""
        if self.T is None:
            raise ValueError("T must be provided.")
        if self.K is None:
            raise ValueError("K must be provided.")
        if float(self.T) <= 0.0:
            raise ValueError("T must be positive.")
        if self._to_scalar(self.K) <= 0.0:
            raise ValueError("K must be positive.")

        upper = float(self.config.integration_upper_bound)
        if upper <= 0.0:
            raise ValueError("integration_upper_bound must be positive.")

        limit = int(self.config.quad_limit)
        if limit <= 0:
            raise ValueError("quad_limit must be positive.")

    def _chf_rn(self, u: complex) -> complex:
        """Evaluate the model characteristic function under the risk-neutral measure."""
        return self.model.chf(chf_input=u, t=float(self.T), risk_neutral=True)

    def _integrand(self, u: float, log_moneyness: float) -> float:
        """Real-valued Lewis integrand evaluated at frequency u for a given log-moneyness."""
        imag_shift = 0.5
        denom = (u * u) + (imag_shift * imag_shift)

        val = np.exp(1j * u * log_moneyness) * self._chf_rn(complex(u, -imag_shift))
        return float(np.real(val) / denom)

    def price(self, S0: Number) -> float:
        """Compute the call price for spot S0 via numerical quadrature of the Lewis integral."""
        self._validate_inputs()

        s0 = self._to_scalar(S0)
        k = self._to_scalar(self.K)

        if s0 <= 0.0:
            raise ValueError("S0 must be positive.")

        log_moneyness = float(np.log(s0 / k))

        integral = quad(
            lambda uu: self._integrand(float(uu), log_moneyness),
            0.0,
            float(self.config.integration_upper_bound),
            limit=int(self.config.quad_limit),
        )[0]

        disc = float(np.exp(-self.risk_free_rate * float(self.T) * 0.5))
        return float(s0 - (1.0 / math.pi) * math.sqrt(s0 * k) * disc * float(integral))

    def price_residual(self, S0: Number) -> float:
        """Return price(S0) minus the configured target price."""
        if self.target_price is None:
            raise ValueError("target_price must be provided to compute a residual.")
        return float(self.price(S0=S0) - self._to_scalar(self.target_price))

    def absolute_price_residual(self, S0: Number) -> float:
        """Return the absolute value of price_residual(S0)."""
        return float(np.abs(self.price_residual(S0=S0)))
