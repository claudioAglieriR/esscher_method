"""
Unit tests for VarianceGamma.parameters_convention_update.

The conventional VG triple (sigma, theta, nu) is a reporting-only view of the
internal (alpha, lambda_P, lambda_M) calibrated parameters. These tests pin
that contract:

  1. After parameters_convention_update() the three keys are present in
     model.parameters.
  2. Their values match paper Eq. 29 exactly (rtol = 1e-12: algebraic
     identities, no statistics or numerical integration).
  3. They are NOT auto-refreshed by a subsequent parameters_update():
     the conventional triple remains stale (= last call to
     parameters_convention_update) until that method is called again.
  4. They are NOT registered in parameter_names or bounds, so check_bounds
     never validates them as calibrated inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from esscher_method.model.model import VarianceGamma
from esscher_method.model.policies import DEFAULT_DELTA


# Algebraic relations are exact: only floating-point round-off bounds the error.
RTOL_ALGEBRAIC = 1e-12


def _vg_with(alpha: float, lambda_p: float, lambda_m: float) -> VarianceGamma:
    """Build a VG model with explicit internal parameters."""
    return VarianceGamma(
        delta=DEFAULT_DELTA,
        parameters={"alpha": float(alpha), "lambda_P": float(lambda_p), "lambda_M": float(lambda_m)},
    )


# Triplets that exercise different regimes:
#   - symmetric (lambda_P = lambda_M, theta = 0)
#   - asymmetric with theta < 0 (heavier left tail, typical of equity)
#   - asymmetric with theta > 0
PARAM_TRIPLETS = [
    pytest.param(50.0, 30.0, 30.0, id="symmetric"),
    pytest.param(120.0, 50.0, 90.0, id="asym_left_skewed"),
    pytest.param(80.0, 60.0, 40.0, id="asym_right_skewed"),
]


@pytest.mark.parametrize("alpha,lambda_p,lambda_m", PARAM_TRIPLETS)
def test_convention_keys_present_after_call(alpha, lambda_p, lambda_m):
    """sigma/theta/nu must appear in parameters after parameters_convention_update()."""
    model = _vg_with(alpha, lambda_p, lambda_m)
    # Keys should not be present before the explicit call.
    assert "sigma" not in model.parameters
    assert "theta" not in model.parameters
    assert "nu" not in model.parameters

    model.parameters_convention_update()

    assert "sigma" in model.parameters
    assert "theta" in model.parameters
    assert "nu" in model.parameters


@pytest.mark.parametrize("alpha,lambda_p,lambda_m", PARAM_TRIPLETS)
def test_convention_values_match_paper_eq29(alpha, lambda_p, lambda_m):
    """
    Eq. 29 of the paper:
        sigma = sqrt(2 * alpha / (lambda_P * lambda_M))
        theta = alpha / lambda_P - alpha / lambda_M
        nu    = 1 / alpha
    The relation is algebraic; the tolerance is round-off only.
    """
    model = _vg_with(alpha, lambda_p, lambda_m)
    model.parameters_convention_update()

    expected_sigma = float(np.sqrt(2.0 * alpha / (lambda_p * lambda_m)))
    expected_theta = float(alpha / lambda_p - alpha / lambda_m)
    expected_nu = float(1.0 / alpha)

    assert np.isclose(model.parameters["sigma"], expected_sigma, rtol=RTOL_ALGEBRAIC, atol=0.0)
    assert np.isclose(model.parameters["theta"], expected_theta, rtol=RTOL_ALGEBRAIC, atol=0.0)
    assert np.isclose(model.parameters["nu"], expected_nu, rtol=RTOL_ALGEBRAIC, atol=0.0)


def test_convention_values_are_stale_after_parameters_update():
    """
    parameters_update() refreshes only the calibrated parameters (alpha,
    lambda_P, lambda_M). The conventional triple must remain at its previous
    value until parameters_convention_update() is called explicitly. This
    pinpoints the documented invariant: sigma/theta/nu are a snapshot, not
    a live view, of the internal state.
    """
    model = _vg_with(alpha=50.0, lambda_p=30.0, lambda_m=30.0)
    model.parameters_convention_update()
    old_sigma = float(model.parameters["sigma"])
    old_theta = float(model.parameters["theta"])
    old_nu = float(model.parameters["nu"])

    # Drive the internal parameters to a different point.
    model.parameters_update(np.asarray([120.0, 70.0, 50.0], dtype=float))

    # The conventional triple must NOT have followed the change.
    assert model.parameters["sigma"] == old_sigma
    assert model.parameters["theta"] == old_theta
    assert model.parameters["nu"] == old_nu

    # Calling the update method explicitly does refresh them.
    model.parameters_convention_update()
    new_alpha, new_lambda_p, new_lambda_m = 120.0, 70.0, 50.0
    expected_sigma = float(np.sqrt(2.0 * new_alpha / (new_lambda_p * new_lambda_m)))
    expected_theta = float(new_alpha / new_lambda_p - new_alpha / new_lambda_m)
    expected_nu = float(1.0 / new_alpha)
    assert np.isclose(model.parameters["sigma"], expected_sigma, rtol=RTOL_ALGEBRAIC, atol=0.0)
    assert np.isclose(model.parameters["theta"], expected_theta, rtol=RTOL_ALGEBRAIC, atol=0.0)
    assert np.isclose(model.parameters["nu"], expected_nu, rtol=RTOL_ALGEBRAIC, atol=0.0)


def test_convention_keys_not_in_calibration_contract():
    """
    sigma/theta/nu must not leak into parameter_names or bounds. The
    invariant enforces 'reporting-only': the calibration pipeline neither
    optimises over them nor validates them via check_bounds.
    """
    model = _vg_with(alpha=50.0, lambda_p=30.0, lambda_m=30.0)
    model.parameters_convention_update()

    for key in ("sigma", "theta", "nu"):
        assert key not in model.parameter_names, (
            f"Conventional key '{key}' must not appear in parameter_names "
            f"(found: {model.parameter_names})."
        )

    # bounds is a list aligned with parameter_names; its length must match
    # the calibrated dimension only.
    assert len(model.bounds) == len(model.parameter_names), (
        f"bounds length ({len(model.bounds)}) must equal len(parameter_names) "
        f"({len(model.parameter_names)})."
    )
