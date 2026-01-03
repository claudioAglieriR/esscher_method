# tests/test_regression_calibration.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esscher_method.calibrator.calibrator import Calibrator
from esscher_method.calibrator.data_calibration import CalibrationData, CalibrationConfig
from esscher_method.model.model import VarianceGamma, BilateralGamma, Merton


# --- Dataset locations (repo-root/data/*)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

MARKET_CSV = DATA_DIR / "data_issuers_2019-2020_market.csv"
DEBT_CSV = DATA_DIR / "data_issuers_2020_debt.csv"

TIME_SPAN = ("2019-10-25", "2020-10-13")
DELTA = 1.0 / 252.0
MATURITY = 1.0


# --- Non-regression baselines (ONLY PD% + parameters)
BASELINES = [
    {
        "ticker": "CRH LN",
        "model": "Merton",
        "pd_pct": 0.003875,
        "params": {"mu": 0.026536, "sigma": 0.303165},
    },
    {
        "ticker": "DG FP",
        "model": "VarianceGamma",
        "pd_pct": 1.058059,
        "params": {"sigma": 0.291066, "theta": -0.189949, "nu": 0.009015},
    },
    {
        "ticker": "FGR FP",
        "model": "BilateralGamma",
        "pd_pct": 3.174467,
        "params": {
            "alpha_M": 118.061748,
            "alpha_P": 149.990569,
            "lambda_M": 95.910583,
            "lambda_P": 132.356921,
        },
    },
]


@pytest.fixture(scope="session")
def datasets():
    """
    Load market and debt datasets once per test session.
    """
    if not MARKET_CSV.exists():
        raise FileNotFoundError(f"Missing market CSV: {MARKET_CSV}")
    if not DEBT_CSV.exists():
        raise FileNotFoundError(f"Missing debt CSV: {DEBT_CSV}")

    market_df = pd.read_csv(MARKET_CSV)
    debt_df = pd.read_csv(DEBT_CSV)

    # Parse and index dates
    if "Dates" not in market_df.columns:
        raise ValueError("Market CSV must contain a 'Dates' column.")
    market_df["Dates"] = pd.to_datetime(market_df["Dates"], dayfirst=True, errors="coerce")
    market_df = market_df.dropna(subset=["Dates"]).set_index("Dates").sort_index()

    if len(debt_df.index) < 1:
        raise ValueError("Debt CSV must contain at least one row.")

    return market_df, debt_df


def _make_model(model_name: str):
    if model_name == "VarianceGamma":
        return VarianceGamma(delta=DELTA)
    if model_name == "BilateralGamma":
        return BilateralGamma(delta=DELTA)
    if model_name == "Merton":
        return Merton(delta=DELTA)
    raise ValueError(f"Unknown model: {model_name}")


def _build_calibration_data(*, market_df: pd.DataFrame, debt_df: pd.DataFrame, ticker: str) -> CalibrationData:
    """
    Build CalibrationData for a ticker using the shared CSV datasets.
    """
    if ticker not in market_df.columns:
        raise KeyError(f"Ticker '{ticker}' not found in market CSV columns.")
    if ticker not in debt_df.columns:
        raise KeyError(f"Ticker '{ticker}' not found in debt CSV columns.")

    equity_series = market_df.loc[TIME_SPAN[0] : TIME_SPAN[1], ticker].dropna().astype(float)
    if equity_series.size < 5:
        raise ValueError(f"Not enough equity observations for ticker '{ticker}' in TIME_SPAN={TIME_SPAN}.")

    debt_value = float(debt_df.loc[0, ticker])

    return CalibrationData(
        equity_values=equity_series,
        debt=debt_value,
        maturity=float(MATURITY),
    )


def _make_calibrator(*, market_df: pd.DataFrame, debt_df: pd.DataFrame, ticker: str, model_name: str) -> Calibrator:
    """
    Create a calibrator with parallelization enabled (thread executor for portability).
    """
    model = _make_model(model_name)

    tolerance = 1e-5 if model_name == "Merton" else 1e-3

    data = _build_calibration_data(market_df=market_df, debt_df=debt_df, ticker=ticker)

    config = CalibrationConfig(
        tolerance=float(tolerance),
        minimization_diff_evolution=True,
        verbose=0,
        max_iterations=20,
        use_parallel=True,
        executor_kind="thread",
        max_workers=1,          
        map_chunksize=1,
        de_seed=12345,
        de_workers=1,
    )

    return Calibrator(model=model, data=data, config=config)


def _assert_dict_close_subset(actual: dict, expected: dict, *, rtol: float, atol: float) -> None:
    """
    Compare expected keys against actual.
    """
    missing = [k for k in expected.keys() if k not in actual]
    assert not missing, f"Missing expected parameter keys: {missing}. Actual keys: {sorted(actual.keys())}"

    for k, v_exp in expected.items():
        v_act = float(actual[k])
        v_exp = float(v_exp)
        assert np.isclose(v_act, v_exp, rtol=rtol, atol=atol), (
            f"Param '{k}' mismatch: got {v_act}, expected {v_exp} (rtol={rtol}, atol={atol})"
        )


@pytest.mark.parametrize("case", BASELINES)
def test_calibration_regression(case, datasets):
    market_df, debt_df = datasets

    cal = _make_calibrator(
        market_df=market_df,
        debt_df=debt_df,
        ticker=case["ticker"],
        model_name=case["model"],
    )

    cal.calibration()

    # Ensure derived parameters exist (VG adds sigma/theta/nu)
    cal.model.parameters_convention_update()

    pd_pct = float(cal.default_probability_computation())
    params = dict(cal.model.parameters)

    # PD regression check
    assert np.isclose(pd_pct, float(case["pd_pct"]), rtol=1e-4, atol=1e-4), (
        f"PD mismatch for {case['ticker']} {case['model']}: got {pd_pct}, expected {case['pd_pct']}"
    )

    # Parameter regression check
    _assert_dict_close_subset(params, case["params"], rtol=1e-4, atol=1e-4)
