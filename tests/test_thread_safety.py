"""
Thread-safety invariant test for AssetInferenceEngine._infer_threaded (P4.T1).

The threaded inference path shares self.model across worker threads. The
contract (pinned by the comment-invariant in asset_inference.py and by these
tests) is:

  model.parameters and model.risk_neutral_parameters are read-only during
  a batch inference. All threads only read; physical and risk-neutral
  parameter updates run on the main thread between batches.

These tests verify the contract by snapshotting both dicts (deep copy)
before infer_asset_values and asserting bit-identical equality afterwards,
across all three Levy models and with max_workers > 1 so multiple threads
are actually engaged.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from esscher_method.calibrator.asset_inference import AssetInferenceEngine
from esscher_method.calibrator.data_calibration import CalibrationConfig, CalibrationData
from esscher_method.model.model import BilateralGamma, Merton, VarianceGamma


DELTA = 1.0 / 252.0

# Parameters close to the CRH LN / 2019-2020 calibration, identical to
# tests/test_integration_convergence.py so the inversion regime is
# representative (asset around 34000, debt around 10500).
MERTON_PARAMS = {"mu": 0.0265, "sigma": 0.3032}
VG_PARAMS = {"alpha": 127.97, "lambda_P": 52.48, "lambda_M": 53.06}
BG_PARAMS = {"alpha_P": 162.76, "lambda_P": 64.49, "alpha_M": 117.99, "lambda_M": 47.25}

# A short equity series is enough to exercise the threaded map; we only need
# more than max_workers entries so that the pool dispatches concurrently.
N_OBS = 32
EQUITY_LEVEL = 33965.0 - 10525.0  # implied equity at CRH LN regime
DEBT = 10525.0
MATURITY = 1.0


def _make_model(name: str):
    if name == "Merton":
        return Merton(delta=DELTA, parameters=dict(MERTON_PARAMS))
    if name == "VarianceGamma":
        return VarianceGamma(delta=DELTA, parameters=dict(VG_PARAMS))
    if name == "BilateralGamma":
        return BilateralGamma(delta=DELTA, parameters=dict(BG_PARAMS))
    raise ValueError(f"Unknown model: {name}")


def _make_engine(model, *, max_workers: int) -> AssetInferenceEngine:
    equity = np.full(N_OBS, EQUITY_LEVEL, dtype=float)
    data = CalibrationData(
        equity_values=equity,
        debt=DEBT,
        maturity=MATURITY,
        risk_free_rate=0.0,
    )
    config = CalibrationConfig(
        use_parallel=True,
        executor_kind="thread",
        max_workers=int(max_workers),
        map_chunksize=1,
        verbose=0,
    )
    return AssetInferenceEngine(model=model, data=data, config=config)


def _assert_dict_bitwise_equal(actual: dict, expected: dict, *, label: str) -> None:
    """Bit-identical equality on dicts of scalars. No tolerance: reads only."""
    assert set(actual.keys()) == set(expected.keys()), (
        f"{label}: key set changed during inference. "
        f"before={sorted(expected.keys())} after={sorted(actual.keys())}"
    )
    for k in expected.keys():
        v_before = expected[k]
        v_after = actual[k]
        # Use == (not isclose): a read-only contract permits zero drift.
        assert v_after == v_before, (
            f"{label}: value for '{k}' changed during inference: "
            f"before={v_before!r} after={v_after!r}"
        )


@pytest.mark.parametrize("model_name", ["Merton", "VarianceGamma", "BilateralGamma"])
@pytest.mark.parametrize("max_workers", [1, 4])
def test_threaded_inference_does_not_mutate_model(model_name: str, max_workers: int):
    """
    After a threaded batch inference, model.parameters and
    model.risk_neutral_parameters must be bit-identical to their pre-batch
    snapshot. Run with max_workers in {1, 4} so the multi-thread path is
    exercised (4 > N_OBS / chunksize would over-parallelise; 4 is enough to
    have several threads compete for self.model reads).
    """
    model = _make_model(model_name)
    engine = _make_engine(model, max_workers=max_workers)

    params_before = copy.deepcopy(model.parameters)
    rn_before = copy.deepcopy(model.risk_neutral_parameters)

    results = engine.infer_asset_values(day_indices=list(range(N_OBS)))

    # Smoke check: inversion returned finite positive values for every day.
    assert len(results) == N_OBS
    arr = np.asarray(results, dtype=float)
    assert np.all(np.isfinite(arr)), f"Non-finite inversion output: {arr}"
    assert np.all(arr > 0.0), f"Non-positive inversion output: {arr}"

    _assert_dict_bitwise_equal(
        dict(model.parameters), params_before, label=f"{model_name} model.parameters"
    )
    _assert_dict_bitwise_equal(
        dict(model.risk_neutral_parameters),
        rn_before,
        label=f"{model_name} model.risk_neutral_parameters",
    )
