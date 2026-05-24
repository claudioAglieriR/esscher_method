"""
Tests for the MartingaleMeasure strategy refactor.

Five contracts are pinned here:

  1. test_esscher_default_reproduces_baseline
     The default Calibrator path (no explicit measure) reproduces the
     regression baseline bit-by-bit at rtol=atol=1e-4. This is the
     primary backward-compatibility tripwire of the refactor.

  2. test_explicit_esscher_equals_default
     Constructing CalibrationConfig(martingale_measure=EsscherMeasure())
     produces the same numerical output as the bare CalibrationConfig().
     Sanity check against typos in the default-resolution logic.

  3. test_mean_correcting_equals_esscher_for_merton
     For Merton (Gaussian), Esscher and mean-correcting EMM coincide
     algebraically (paper Remark 3 / Eq. 15: mu_RN = r - sigma^2/2
     regardless of method). The test pins this identity at rtol=1e-8.
     

  4. test_mean_correcting_unimplemented_on_bg_vg
     MeanCorrectingMeasure raises NotImplementedError with a clear
     message on BG / VG, surfacing the prototype scope to callers.

  5. test_custom_emm_is_injectable
     A user-defined MartingaleMeasure subclass plugged into
     CalibrationConfig is actually invoked by the Calibrator's
     risk_neutral_update, paired solve / apply.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from esscher_method.calibrator.calibrator import Calibrator
from esscher_method.calibrator.data_calibration import CalibrationConfig, CalibrationData
from esscher_method.calibrator.martingale_measure import (
    EsscherMeasure,
    MartingaleMeasure,
    MeanCorrectingMeasure,
)
from esscher_method.model.model import BilateralGamma, Merton, Model, VarianceGamma


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MARKET_CSV = DATA_DIR / "data_issuers_2019-2020_market.csv"
DEBT_CSV = DATA_DIR / "data_issuers_2020_debt.csv"

TIME_SPAN = ("2019-10-25", "2020-10-13")
DELTA = 1.0 / 252.0
MATURITY = 1.0

# Baseline used by tests 1 / 2 / 3 / 5: CRH LN / Merton at r=0 (fastest case
# in the regression matrix). Pinpoint values are mirrored from
# tests/test_regression_calibration.py::BASELINES.
BASELINE_TICKER = "CRH LN"
BASELINE_PD_PCT = 0.003875
BASELINE_PARAMS = {"mu": 0.026536, "sigma": 0.303165}


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


def _make_calibration_data(market_df: pd.DataFrame, debt_df: pd.DataFrame, ticker: str) -> CalibrationData:
    equity_series = market_df.loc[TIME_SPAN[0] : TIME_SPAN[1], ticker].dropna().astype(float)
    debt_value = float(debt_df.loc[0, ticker])
    return CalibrationData(
        equity_values=equity_series,
        debt=debt_value,
        maturity=MATURITY,
        risk_free_rate=0.0,
    )


def _make_config(
    *,
    martingale_measure=None,
    max_iterations: int = 20,
    tolerance: float = 1e-5,
) -> CalibrationConfig:
    """Single source of truth for the deterministic config used by the tests."""
    return CalibrationConfig(
        tolerance=tolerance,
        minimization_diff_evolution=True,
        verbose=0,
        max_iterations=max_iterations,
        use_parallel=True,
        executor_kind="thread",
        max_workers=1,
        map_chunksize=1,
        de_seed=12345,
        de_workers=1,
        martingale_measure=martingale_measure,
    )


def _run_merton(
    datasets,
    *,
    martingale_measure=None,
    max_iterations: int = 20,
) -> Dict[str, object]:
    market_df, debt_df = datasets
    data = _make_calibration_data(market_df, debt_df, BASELINE_TICKER)
    config = _make_config(martingale_measure=martingale_measure, max_iterations=max_iterations)
    model = Merton(delta=DELTA)
    cal = Calibrator(model=model, data=data, config=config)
    cal.calibration()
    return {
        "calibrator": cal,
        "pd_pct": float(cal.default_probability_computation()),
        "params": {k: float(cal.model.parameters[k]) for k in cal.model.parameter_names},
        "p_star": float(cal.p_star),
    }


@pytest.fixture(scope="module")
def default_merton_result(datasets):
    """One default-path Merton calibration shared by tests 1 and 2."""
    return _run_merton(datasets, martingale_measure=None)


# ----------------------------- Test 1 -----------------------------------------

def test_esscher_default_reproduces_baseline(default_merton_result):
    """
    Primary tripwire: default CalibrationConfig() must reproduce the regression
    baseline bit-by-bit at the same tolerance the regression itself uses.
    A failure here means the dispatch refactor altered the default Esscher path.
    """
    result = default_merton_result
    assert np.isclose(result["pd_pct"], BASELINE_PD_PCT, rtol=1e-4, atol=1e-4), (
        f"Default path PD drifted from baseline: got {result['pd_pct']}, "
        f"expected {BASELINE_PD_PCT}"
    )
    for k, v_exp in BASELINE_PARAMS.items():
        v_got = result["params"][k]
        assert np.isclose(v_got, v_exp, rtol=1e-4, atol=1e-4), (
            f"Default path parameter '{k}' drifted: got {v_got}, expected {v_exp}"
        )


# ----------------------------- Test 2 -----------------------------------------

def test_explicit_esscher_equals_default(datasets, default_merton_result):
    """
    Passing martingale_measure=EsscherMeasure() explicitly must produce
    identical output to the default. Both paths go through the same EsscherMeasure
    code (the default resolution builds the same object); equality is bitwise
    on PD and parameters.
    """
    explicit = _run_merton(datasets, martingale_measure=EsscherMeasure())

    # Bitwise on PD: both paths run the exact same EsscherMeasure(config=None).
    assert explicit["pd_pct"] == default_merton_result["pd_pct"], (
        f"Explicit EsscherMeasure() diverged from default: "
        f"explicit_pd={explicit['pd_pct']}, default_pd={default_merton_result['pd_pct']}"
    )
    for k in BASELINE_PARAMS.keys():
        assert explicit["params"][k] == default_merton_result["params"][k], (
            f"Explicit EsscherMeasure() parameter '{k}' diverged: "
            f"explicit={explicit['params'][k]}, default={default_merton_result['params'][k]}"
        )

    # p_star is mirrored in both paths (both use EsscherMeasure).
    assert np.isfinite(explicit["p_star"])
    assert explicit["p_star"] == default_merton_result["p_star"]


# ----------------------------- Test 3 -----------------------------------------

def test_mean_correcting_equals_esscher_for_merton(datasets, default_merton_result):
    """
    For Merton (Gaussian), Esscher and mean-correcting EMM coincide
    algebraically: both give mu_RN = r - sigma^2/2 (paper Remark 3 / Eq. 15).
    Pin the identity within rtol=1e-8: brentq solver jitter on p_star
    (~1e-12) propagates to mu_RN as sigma^2 * jitter ~ 1e-13, well below
    1e-8 even after accumulated calibration iterations.

    The EMM arbitrariness is zero on Gaussian processes. The arbitrariness becomes real on
    BG / VG, which the prototype declines to handle (see test 4).

    p_star is NaN on the mean-correcting branch (it is Esscher-specific),
    so it is not part of the equality check.
    """
    meancorr = _run_merton(datasets, martingale_measure=MeanCorrectingMeasure())

    # PD agreement: algebraic identity at solver-precision tolerance.
    assert np.isclose(meancorr["pd_pct"], default_merton_result["pd_pct"], rtol=1e-8, atol=1e-8), (
        f"Esscher and MeanCorrecting diverged on Merton PD: "
        f"meancorr={meancorr['pd_pct']}, esscher={default_merton_result['pd_pct']} "
        f"(rtol=1e-8). They must coincide algebraically; see paper Remark 3."
    )
    for k in BASELINE_PARAMS.keys():
        assert np.isclose(
            meancorr["params"][k],
            default_merton_result["params"][k],
            rtol=1e-8,
            atol=1e-8,
        ), (
            f"Esscher and MeanCorrecting diverged on Merton parameter '{k}': "
            f"meancorr={meancorr['params'][k]}, esscher={default_merton_result['params'][k]}"
        )

    # p_star must be NaN on the mean-correcting branch (Esscher-specific).
    assert np.isnan(meancorr["p_star"]), (
        f"MeanCorrecting path leaked a non-NaN p_star: {meancorr['p_star']}. "
        f"p_star is Esscher-specific and must be NaN under other measures."
    )


# ----------------------------- Test 4 -----------------------------------------

@pytest.mark.parametrize(
    "model_name,ticker",
    [
        ("VarianceGamma", "DG FP"),
        ("BilateralGamma", "FGR FP"),
    ],
)
def test_mean_correcting_unimplemented_on_bg_vg(datasets, model_name, ticker):
    """
    The prototype scope of MeanCorrectingMeasure is Merton only. For BG and VG
    the user must get a clear NotImplementedError with a message that explains
    the scope and recommends EsscherMeasure.

    The error must surface from the first risk_neutral_update inside
    historical_values_init (after the bootstrap physical fit). The
    Calibrator must NOT wrap the NotImplementedError in a generic
    ValueError (which would mask the original cause).
    """
    market_df, debt_df = datasets
    data = _make_calibration_data(market_df, debt_df, ticker)
    config = _make_config(martingale_measure=MeanCorrectingMeasure())
    if model_name == "VarianceGamma":
        model: Model = VarianceGamma(delta=DELTA)
    elif model_name == "BilateralGamma":
        model = BilateralGamma(delta=DELTA)
    else:
        raise ValueError(model_name)

    cal = Calibrator(model=model, data=data, config=config)
    with pytest.raises(NotImplementedError, match=r"MeanCorrectingMeasure currently implements only Merton"):
        cal.calibration()


# ----------------------------- Test 5 -----------------------------------------

class _CountingMeasure(MartingaleMeasure):
    """Local mock that counts solve / apply calls; returns a no-tilt Esscher result."""

    def __init__(self) -> None:
        self.solve_calls = 0
        self.apply_calls = 0
        self.last_p_initial: float = float("nan")

    def solve(self, *, model, risk_free_rate, delta, p_initial: float = 0.0) -> float:
        self.solve_calls += 1
        self.last_p_initial = float(p_initial)
        # Return 0.0: for Merton this gives mu_RN = mu, sigma_RN = sigma.
        # The downstream pricer still runs (RN coincides with physical drift
        # but the loop exercises the dispatch end-to-end).
        return 0.0

    def apply(self, *, model, emm_param: float) -> Dict[str, float]:
        self.apply_calls += 1
        return dict(model.risk_neutral_parameters_update(p_star=float(emm_param)))


def test_custom_emm_is_injectable(datasets):
    """
    A user-defined MartingaleMeasure subclass plugged into CalibrationConfig
    must be invoked by the Calibrator's risk_neutral_update. Verify:
      - Calibrator stores the same instance (no defensive copy).
      - solve() and apply() are both called at least once (dispatch fired).
      - solve() and apply() are paired (same count): every solve is followed
        by an apply; no path silently bypasses one of them.
      - p_star is NaN (the custom measure is not Esscher).
    """
    measure = _CountingMeasure()
    # max_iterations=3 keeps the runtime small; the dispatch is exercised
    # at iteration 0 (historical_values_init) plus the recurrent loop.
    result = _run_merton(datasets, martingale_measure=measure, max_iterations=3)
    cal = result["calibrator"]

    assert cal.martingale_measure is measure, (
        "Calibrator must store the injected MartingaleMeasure by reference, "
        "not a copy."
    )
    assert measure.solve_calls >= 1, "Custom measure solve() was never invoked."
    assert measure.apply_calls >= 1, "Custom measure apply() was never invoked."
    assert measure.solve_calls == measure.apply_calls, (
        f"solve/apply pairing broken: solve_calls={measure.solve_calls}, "
        f"apply_calls={measure.apply_calls}. Every solve must be followed by an apply."
    )
    assert np.isnan(result["p_star"]), (
        f"Custom (non-Esscher) measure must produce NaN p_star, got {result['p_star']}."
    )
