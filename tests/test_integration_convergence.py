"""
Convergence tests for the Fourier-related numerical integrations used in the package.

These tests verify that the default values of:
- LewisPricerConfig.quad_limit
- IntegrationConfig.cdf_quad_limit

are sufficient: increasing them by 4x does not change the integration result
beyond a strict tolerance (1e-8). This makes the default settings a verifiable
contract and protects against silent regressions if the defaults are tuned again.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from esscher_method.calibrator.pricer import LewisEuropeanTargetPricer, LewisPricerConfig
from esscher_method.model.model import BilateralGamma, Merton, VarianceGamma
from esscher_method.model.numerics import IntegrationConfig


DELTA = 1.0 / 252.0
CONVERGENCE_TOL = 1e-8

# Representative parameters per model, close to the CRH LN / 2019-2020 calibration.
MERTON_PARAMS = {"mu": 0.0265, "sigma": 0.3032}
VG_PARAMS = {"alpha": 127.97, "lambda_P": 52.48, "lambda_M": 53.06}
BG_PARAMS = {"alpha_P": 162.76, "lambda_P": 64.49, "alpha_M": 117.99, "lambda_M": 47.25}

# Representative pricing inputs (CRH LN: implied asset around 34000, debt 10525).
S0 = 33965.0
K = 10525.0
T = 1.0


def _make_model(name: str):
    if name == "Merton":
        return Merton(delta=DELTA, parameters=dict(MERTON_PARAMS))
    if name == "VarianceGamma":
        return VarianceGamma(delta=DELTA, parameters=dict(VG_PARAMS))
    if name == "BilateralGamma":
        return BilateralGamma(delta=DELTA, parameters=dict(BG_PARAMS))
    raise ValueError(f"Unknown model: {name}")


@pytest.mark.parametrize("model_name", ["Merton", "VarianceGamma", "BilateralGamma"])
def test_lewis_pricer_quad_limit_is_converged(model_name: str) -> None:
    """Lewis Fourier integral at the default quad_limit matches a 4x-tighter setting within 1e-8."""
    model = _make_model(model_name)

    cfg_default = LewisPricerConfig()
    cfg_strict = dataclasses.replace(cfg_default, quad_limit=cfg_default.quad_limit * 4)

    price_default = LewisEuropeanTargetPricer(
        model=model, T=T, K=K, risk_free_rate=0.0, config=cfg_default
    ).price(S0=S0)
    price_strict = LewisEuropeanTargetPricer(
        model=model, T=T, K=K, risk_free_rate=0.0, config=cfg_strict
    ).price(S0=S0)

    diff = abs(price_default - price_strict)
    assert diff < CONVERGENCE_TOL, (
        f"{model_name}: Lewis price not converged at default quad_limit "
        f"(default={price_default}, strict={price_strict}, diff={diff})"
    )


@pytest.mark.parametrize("model_name", ["Merton", "VarianceGamma", "BilateralGamma"])
def test_gil_pelaez_cdf_quad_limit_is_converged(model_name: str) -> None:
    """Gil-Pelaez CDF at the default cdf_quad_limit matches a 4x-tighter setting within 1e-8."""
    model = _make_model(model_name)
    x = -math.log(S0 / K)
    t = T

    cfg_default = IntegrationConfig()
    cfg_strict = dataclasses.replace(cfg_default, cdf_quad_limit=cfg_default.cdf_quad_limit * 4)

    cdf_default = model.cdf(cdf_input=x, t=t, config=cfg_default)
    cdf_strict = model.cdf(cdf_input=x, t=t, config=cfg_strict)

    diff = abs(cdf_default - cdf_strict)
    assert diff < CONVERGENCE_TOL, (
        f"{model_name}: CDF not converged at default cdf_quad_limit "
        f"(default={cdf_default}, strict={cdf_strict}, diff={diff})"
    )
