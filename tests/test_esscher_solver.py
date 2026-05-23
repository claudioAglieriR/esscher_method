"""
Unit tests for EsscherSolver.

These tests target the solver in isolation, without running the full calibration
pipeline. The first test verifies that the bounded least-squares fallback rejects
a candidate that is not a true root of the Esscher equation, instead of silently
propagating a local minimum of |residual|.
"""
from __future__ import annotations

import pytest

from esscher_method.calibrator.esscher_solver import EsscherSolver, EsscherSolverConfig
from esscher_method.model.model import BilateralGamma


DELTA = 1.0 / 252.0


class FlatCGFModel:
    """
    Minimal model with a constant CGF: K(p) = 0 for all p.

    Consequences for the Esscher equation K(p+1) - K(p) = r*delta:
      - if r != 0, the equation has NO root and the solver must raise.
      - if r == 0, the equation reduces to 0 = 0 and any p in bounds is a trivial root.

    This geometry forces the failure mode that the post-fallback residual check
    guards against: the grid bracketing fails (constant residual has no sign change),
    and the bounded least-squares fallback converges to a local minimum where
    |residual| = |r*delta|, which is NOT a root when r != 0. Without the residual
    check, the solver would return this candidate silently.
    """

    def esscher_p_star_bounds(self):
        return (-1.0, 1.0)

    def cumulant_generating_function(self, cgf_input):
        return 0.0


def test_solver_rejects_non_root_in_fallback():
    """Flat CGF with r > 0 has no Esscher root; the solver must raise instead of silently returning a non-root."""
    model = FlatCGFModel()
    solver = EsscherSolver(model=model, risk_free_rate=0.05, delta=DELTA)

    with pytest.raises(ValueError, match="did not converge to a root"):
        solver.solve()


def test_solver_accepts_trivial_root_with_zero_rate():
    """Flat CGF with r = 0 satisfies the Esscher equation trivially; the solver must not raise."""
    model = FlatCGFModel()
    solver = EsscherSolver(model=model, risk_free_rate=0.0, delta=DELTA)

    p_star = solver.solve()
    assert isinstance(p_star, float)
    assert -1.0 <= p_star <= 1.0


def test_solver_happy_path_on_bg_model():
    """A well-posed BG model with reasonable parameters returns a finite p_star."""
    model = BilateralGamma(
        delta=DELTA,
        parameters={"alpha_P": 150.0, "lambda_P": 70.0, "alpha_M": 100.0, "lambda_M": 50.0},
    )
    solver = EsscherSolver(model=model, risk_free_rate=0.03, delta=DELTA)

    p_star = solver.solve()
    assert isinstance(p_star, float)
    lower, upper = model.esscher_p_star_bounds()
    assert lower <= p_star <= upper


def test_solver_residual_tol_is_configurable():
    """The residual_tol field on EsscherSolverConfig propagates to the rejection threshold."""
    model = FlatCGFModel()
    very_loose = EsscherSolverConfig(residual_tol=1.0)
    solver_loose = EsscherSolver(
        model=model, risk_free_rate=0.05, delta=DELTA, config=very_loose
    )
    # With tolerance 1.0, the residual |r*delta| ~ 2e-4 is well within tolerance, so accepted.
    p_star = solver_loose.solve()
    assert isinstance(p_star, float)
