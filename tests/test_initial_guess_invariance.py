"""
Seed-invariance test for the equity-cumulants bootstrap.

historical_values_init() bootstraps the calibration loop with a seed
obtained by fitting physical parameters on equity log-returns. The
docstring of that method states that the seed is corrected by the loop
in recurrent_estimation() from iteration 1 onward. This test pins that
claim empirically: running the calibration with two markedly different
in-bounds Merton seeds must converge to the same point (physical
parameters and PD) within rtol = 1e-3 when max_iterations >= 30 and
the paper-default DE-enabled optimiser is used.

This protects against future "principled seed" refactors that would
silently shift the convergence point under finite max_iterations and
break the paper baseline (see CLAUDE.md, Strategic objective: Default
Stabile).

Scope: Merton only. Merton is the simplest model (2 parameters), keeps
the runtime in the default layer, and is sufficient to document the
principle. BG and VG would only add runtime without adding information
to the empirical claim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esscher_method.calibrator.calibrator import Calibrator
from esscher_method.calibrator.data_calibration import CalibrationConfig, CalibrationData
from esscher_method.model.model import Merton


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MARKET_CSV = DATA_DIR / "data_issuers_2019-2020_market.csv"
DEBT_CSV = DATA_DIR / "data_issuers_2020_debt.csv"

TIME_SPAN = ("2019-10-25", "2020-10-13")
DELTA = 1.0 / 252.0
MATURITY = 1.0
TICKER = "CRH LN"

# Two seeds that differ by orders of magnitude on both Merton parameters
# but stay well inside the policy bounds (mu in [-4, 4], sigma in [1e-8, 4]).
# Seed A is the policy default; seed B is deliberately far in the opposite
# direction on mu and an order of magnitude larger on sigma.
SEED_DEFAULT = {"mu": 1.0, "sigma": 1e-2}
SEED_FAR = {"mu": -2.0, "sigma": 1.5}

MAX_ITERATIONS = 30
TOLERANCE = 1e-5
# Empirical agreement target between the two converged calibrations.
# 1e-3 follows the spec; PD is in percent so we use absolute too.
RTOL_CONVERGENCE = 1e-3
ATOL_CONVERGENCE = 1e-3


@pytest.fixture(scope="module")
def datasets():
    if not MARKET_CSV.exists():
        raise FileNotFoundError(f"Missing market CSV: {MARKET_CSV}")
    if not DEBT_CSV.exists():
        raise FileNotFoundError(f"Missing debt CSV: {DEBT_CSV}")

    market_df = pd.read_csv(MARKET_CSV)
    market_df["Dates"] = pd.to_datetime(market_df["Dates"], dayfirst=True, errors="coerce")
    market_df = market_df.dropna(subset=["Dates"]).set_index("Dates").sort_index()
    debt_df = pd.read_csv(DEBT_CSV)
    return market_df, debt_df


def _build_calibrator(market_df, debt_df, seed_params: dict) -> Calibrator:
    equity_series = market_df.loc[TIME_SPAN[0] : TIME_SPAN[1], TICKER].dropna().astype(float)
    debt_value = float(debt_df.loc[0, TICKER])
    data = CalibrationData(
        equity_values=equity_series,
        debt=debt_value,
        maturity=MATURITY,
        risk_free_rate=0.0,
    )
    # Paper-default optimiser configuration (DE on, single-threaded for
    # determinism, identical to tests/test_regression_calibration.py except
    # for max_iterations bumped to MAX_ITERATIONS).
    config = CalibrationConfig(
        tolerance=TOLERANCE,
        minimization_diff_evolution=True,
        verbose=0,
        max_iterations=MAX_ITERATIONS,
        use_parallel=True,
        executor_kind="thread",
        max_workers=1,
        map_chunksize=1,
        de_seed=12345,
        de_workers=1,
    )
    model = Merton(delta=DELTA, parameters=dict(seed_params))
    return Calibrator(model=model, data=data, config=config)


def test_merton_calibration_is_invariant_to_initial_seed(datasets):
    """
    Two Merton calibrations on the same data with markedly different
    seeds must converge to the same (mu, sigma) and PD within rtol=1e-3.
    Pins the docstring claim of historical_values_init: the equity seed
    is corrected by the loop and the converged point is seed-independent.
    """
    market_df, debt_df = datasets

    cal_a = _build_calibrator(market_df, debt_df, SEED_DEFAULT)
    cal_a.calibration()
    params_a = {k: float(cal_a.model.parameters[k]) for k in cal_a.model.parameter_names}
    pd_a = float(cal_a.default_probability_computation())

    cal_b = _build_calibrator(market_df, debt_df, SEED_FAR)
    cal_b.calibration()
    params_b = {k: float(cal_b.model.parameters[k]) for k in cal_b.model.parameter_names}
    pd_b = float(cal_b.default_probability_computation())

    # Sanity: both runs must have converged (not exited on max_iterations).
    # If max_iterations was hit the comparison below is uninformative; flag it.
    assert int(cal_a.iterations_number) < MAX_ITERATIONS, (
        f"Seed A hit max_iterations={MAX_ITERATIONS}; convergence not reached."
    )
    assert int(cal_b.iterations_number) < MAX_ITERATIONS, (
        f"Seed B hit max_iterations={MAX_ITERATIONS}; convergence not reached."
    )

    # Parameter agreement: each calibrated parameter coincides within rtol=1e-3.
    for name in cal_a.model.parameter_names:
        v_a = params_a[name]
        v_b = params_b[name]
        assert np.isclose(v_a, v_b, rtol=RTOL_CONVERGENCE, atol=ATOL_CONVERGENCE), (
            f"Parameter '{name}' diverges across seeds: "
            f"seed_default={v_a}, seed_far={v_b} (rtol={RTOL_CONVERGENCE})"
        )

    # PD agreement: the headline output also coincides within rtol=1e-3.
    # PD is expressed in percent, so atol=1e-3 corresponds to 0.001 percentage
    # points (i.e. 10 basis points of absolute PD).
    assert np.isclose(pd_a, pd_b, rtol=RTOL_CONVERGENCE, atol=ATOL_CONVERGENCE), (
        f"PD diverges across seeds: seed_default={pd_a}, seed_far={pd_b} "
        f"(rtol={RTOL_CONVERGENCE}, atol={ATOL_CONVERGENCE} pct points)"
    )
