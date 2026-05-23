"""
Unit tests for the relative-tolerance convergence criterion used by Calibrator.

These tests inject a synthetic physical-history into a Calibrator and verify that
_physical_parameters_converged behaves as expected on edge cases the relative
form is designed to handle (small-magnitude parameters oscillating while
large-magnitude parameters appear stable, division by zero on near-zero
previous values).
"""
from __future__ import annotations

import numpy as np

from esscher_method.calibrator.calibrator import Calibrator
from esscher_method.calibrator.data_calibration import CalibrationConfig, CalibrationData
from esscher_method.model.model import Merton


N_OBS = 252
DELTA = 1.0 / 252.0


def _make_minimal_calibrator() -> Calibrator:
    """Create a Calibrator with valid-but-unused state, just to exercise its convergence method."""
    equity = np.linspace(100.0, 110.0, N_OBS)
    data = CalibrationData(equity_values=equity, debt=50.0, maturity=1.0)
    config = CalibrationConfig(use_parallel=False, verbose=0)
    return Calibrator(model=Merton(delta=DELTA), data=data, config=config)


def test_relative_tolerance_catches_oscillating_small_parameter():
    """
    A small-magnitude parameter oscillating by ~50% relative is NOT converged
    under the relative criterion, even at tolerance 1e-3. An absolute criterion
    would falsely declare convergence because the absolute step (~5e-4) is below 1e-3.
    """
    cal = _make_minimal_calibrator()
    cal.physical_history = [
        {"mu": 0.0010, "sigma": 100.000001},
        {"mu": 0.0015, "sigma": 100.000002},
    ]
    # mu: relative diff ~ 50 %, sigma: relative diff ~ 1e-8. The mu coordinate
    # is the one that must trigger the False verdict.
    assert cal._physical_parameters_converged(tolerance=1e-3) is False


def test_relative_tolerance_accepts_uniformly_stable_parameters():
    """All parameters stable below the relative tolerance: converged."""
    cal = _make_minimal_calibrator()
    cal.physical_history = [
        {"mu": 0.001000, "sigma": 100.0000},
        {"mu": 0.001001, "sigma": 100.0001},  # both ~ 1e-6 relative
    ]
    assert cal._physical_parameters_converged(tolerance=1e-3) is True


def test_single_iteration_history_is_never_converged():
    """A history with fewer than two recorded iterations cannot be converged."""
    cal = _make_minimal_calibrator()
    cal.physical_history = [{"mu": 0.0, "sigma": 1.0}]
    assert cal._physical_parameters_converged(tolerance=1e-3) is False


def test_zero_previous_value_uses_small_denominator_floor():
    """
    When |previous| is below 1e-8, the relative criterion uses 1e-8 as denominator
    floor to avoid division by zero. A tiny absolute step well below
    tolerance * 1e-8 stays converged.
    """
    cal = _make_minimal_calibrator()
    cal.physical_history = [
        {"mu": 0.0, "sigma": 1.0},
        {"mu": 1e-12, "sigma": 1.000001},
    ]
    # mu absolute diff = 1e-12, floor denom = 1e-8, relative = 1e-4 < tol.
    # sigma relative ~ 1e-6 < tol. Both converged.
    assert cal._physical_parameters_converged(tolerance=1e-3) is True
