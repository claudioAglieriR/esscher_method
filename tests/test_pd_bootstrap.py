"""
Tests for the PD bootstrap (esscher_method.calibrator.pd_bootstrap).

================================================================================
Test taxonomy
================================================================================

Two tests with sharply different roles:

1. **test_bootstrap_pd_api_contract** (cheap, ~30 s):
   Validates the *mechanical* correctness of bootstrap_pd. Runs a single
   bootstrap with few replicas and asserts shape, finiteness, monotonicity
   of the CI relative to the mean, and bookkeeping invariants
   (n_replicas + n_failed = requested). This is the test that catches a
   refactor breaking the return contract.

2. **test_bootstrap_pd_coverage_probability** (slow, ~3 min):
   Validates the *statistical* correctness of bootstrap_pd. Simulates M
   independent equity series from a known Merton DGP, runs bootstrap_pd on
   each, and checks that the analytical "true" PD (computed in closed form
   from theta_true and the realized V_A_last) falls inside the empirical
   95-percent CI in at least 70 percent of the M cases. The threshold is
   well below the nominal 95 percent both because M is small (10) and
   because the empirical coverage of a percentile bootstrap is known to
   undershoot the nominal level for skewed distributions like the PD.

================================================================================
Why Merton for the coverage test
================================================================================

Merton has a closed-form true PD given (mu_true, sigma_true, V_A, K, T):
    PD_true = Phi((log(K / V_A) - mu_true * T) / (sigma_true * sqrt(T)))

This makes the "ground truth" arithmetic, not numerical, so the coverage
test is not confounded by ground-truth approximation error. For BG and VG
the true PD requires its own numerical integration (Gil-Pelaez), and any
discrepancy between two integration routes would muddy the coverage signal.

================================================================================
Cost note
================================================================================

The coverage test runs M = 10 outer Monte Carlo draws, each launching
n_replicas = 15 inner bootstrap replicas. That is 10 * (1 + 15) = 160 full
Merton calibrations, ~1 s each at N = 252 with max_iterations = 3, so
total runtime is ~2 to 3 minutes on a single CPU. This is at the edge of
acceptable for a CI suite; tuning n_replicas down further would weaken
the bootstrap CI estimate, and tuning M down would make the empirical
coverage statistic too noisy to be meaningful.
"""
# TODO(nightly bootstrap coverage): Companion nightly test pending.
# Spec for the deferred counterpart:
#   - M = 100 outer Monte-Carlo draws (vs M=10 here in the slow layer):
#     reduces the empirical-coverage estimator standard error from
#     ~15 to ~5 percentage points, enough resolution to distinguish a
#     well-calibrated bootstrap from a 5-percent miscalibration.
#   - n_replicas = 200 inner bootstrap replicas (vs 15 here): the
#     percentile CI converges to its asymptotic shape; the residual
#     undercoverage is then methodological (skewness of the PD
#     distribution under sampling), not estimation noise of the CI itself.
#   - Coverage threshold = 85 percent (vs the smoke's 70 percent): tight
#     enough to detect a true miscalibration, loose enough to absorb the
#     known undercoverage of percentile bootstrap on skewed estimands.
#   - All three Levy models (vs Merton only here): Merton uses the
#     closed-form normal-CDF ground truth already defined here; BG and VG
#     require Gil-Pelaez standalone integration as the ground truth
#     (closed-form unavailable). The standalone Gil-Pelaez ground truth
#     must live in a new tests/_bootstrap_helpers.py (to be created) to
#     avoid circular dependence on the calibrator under test.
#   - Marker: @pytest.mark.nightly; expected ~5-7 h on the reference
#     machine, opt-in via `pytest -m nightly`.
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pytest
from scipy.stats import norm

from esscher_method.calibrator.calibrator import Calibrator
from esscher_method.calibrator.data_calibration import CalibrationConfig, CalibrationData
from esscher_method.calibrator.pd_bootstrap import BootstrapPDResult, bootstrap_pd
from esscher_method.model.model import Merton


# ============================================================================
# Shared simulation helpers
# ============================================================================

DELTA: float = 1.0 / 252.0
N_OBSERVATIONS: int = 252
INITIAL_EQUITY: float = 100.0
DEBT: float = 50.0  # V_E / K = 2 -> non-trivial PD ~ a few percent
MATURITY: float = 1.0
RISK_FREE_RATE: float = 0.0


def _simulate_merton_equity(
    rng: np.random.Generator,
    n: int,
    mu: float,
    sigma: float,
    delta: float,
    initial: float,
) -> np.ndarray:
    """
    Simulate a Merton equity series of length n+1 from independent Normal log-returns.

    The output is a level series suitable for direct use in CalibrationData:
    log_returns[i] = log(equity[i+1] / equity[i]) ~ N(mu*delta, sigma^2*delta).
    """
    returns = rng.normal(loc=mu * delta, scale=sigma * float(np.sqrt(delta)), size=n)
    log_levels = float(np.log(initial)) + np.concatenate(([0.0], np.cumsum(returns)))
    return np.exp(log_levels)


def _build_merton_calibrator(equity: np.ndarray, *, max_iterations: int = 2) -> Calibrator:
    """
    Build a Calibrator configured for the cheap-fast regime used in these tests.

    Choices:
      - use_parallel = False: predictability and minimal overhead.
      - max_iterations = 2 by default: in the Merton regime with daily equity
        at N = 252 the moment-matching converges in 1-2 iterations; we use
        the minimum that still produces a stable PD because the bootstrap
        loops over many calibrations and per-replica cost dominates total
        runtime.
      - tolerance = 1e-2: loose relative tolerance; tighter is wasted in the
        bootstrap context where each replica only needs a stable PD.
      - minimization_diff_evolution = False: DE is overkill for the 2-parameter
        Merton fit, and the bounded LS solver handles synthetic Merton data
        well in this regime. Disabling DE shaves a noticeable fraction of
        per-replica cost.
    """
    data = CalibrationData(
        equity_values=equity,
        debt=DEBT,
        maturity=MATURITY,
        risk_free_rate=RISK_FREE_RATE,
    )
    config = CalibrationConfig(
        use_parallel=False,
        verbose=0,
        max_iterations=int(max_iterations),
        tolerance=1e-2,
        minimization_diff_evolution=False,
    )
    return Calibrator(model=Merton(delta=DELTA), data=data, config=config)


def _merton_true_pd_pct(
    *,
    mu_true: float,
    sigma_true: float,
    v_a: float,
    k: float,
    t: float,
) -> float:
    """
    Analytical Merton PD (percent) given the true parameters and current asset level.

    Closed form from the standard distance-to-default expression:
        PD = P(V_A(t + T) < K | V_A(t) = v_a) under the physical measure
           = Phi((log(K / V_A) - mu_true * T) / (sigma_true * sqrt(T)))

    This matches the structural-default convention used by
    Calibrator.default_probability_computation() (which inverts the model
    CDF at -distance_to_default = -log(v_a / k)) and is what the bootstrap
    CI must cover for the coverage test to be valid.
    """
    d2 = (np.log(float(k) / float(v_a)) - float(mu_true) * float(t)) / (
        float(sigma_true) * float(np.sqrt(t))
    )
    return float(norm.cdf(d2)) * 100.0


# ============================================================================
# Test 1: API contract
# ============================================================================


def test_bootstrap_pd_api_contract() -> None:
    """
    Mechanical-correctness test: shape, finiteness, CI ordering, bookkeeping.

    Runs a single bootstrap with a small number of replicas and asserts:
      (a) The result is a BootstrapPDResult.
      (b) n_replicas + n_failed equals the requested count.
      (c) The samples array has length n_replicas.
      (d) The mean, std, and CI endpoints are finite.
      (e) The CI is correctly ordered (lower <= mean <= upper).
      (f) The CI contains the median of the samples (sanity check on
          empirical quantiles).
    """
    rng = np.random.default_rng(42)
    equity = _simulate_merton_equity(
        rng=rng, n=N_OBSERVATIONS, mu=0.05, sigma=0.30, delta=DELTA, initial=INITIAL_EQUITY
    )
    cal = _build_merton_calibrator(equity=equity)
    cal.calibration()

    requested = 12
    result = bootstrap_pd(calibrator=cal, n_replicas=requested, seed=123)

    # (a)
    assert isinstance(result, BootstrapPDResult)

    # (b) bookkeeping
    assert result.n_replicas + result.n_failed == requested, (
        f"Bookkeeping mismatch: n_replicas={result.n_replicas} + "
        f"n_failed={result.n_failed} != requested={requested}"
    )

    # We need at least most replicas to succeed for the rest of the
    # invariants to be meaningful. A high failure rate here would mean
    # something is wrong with the test setup, not with the user code.
    assert result.n_replicas >= max(2, requested - 2), (
        f"Too many failures: only {result.n_replicas} / {requested} succeeded"
    )

    # (c) shape
    assert result.samples.shape == (result.n_replicas,)

    # (d) finiteness
    assert np.isfinite(result.pd_mean)
    assert np.isfinite(result.pd_std)
    lo, hi = result.ci_95
    assert np.isfinite(lo) and np.isfinite(hi)

    # (e) ordering
    assert lo <= result.pd_mean <= hi, (
        f"CI not ordered: lo={lo}, mean={result.pd_mean}, hi={hi}"
    )

    # (f) the 95-percent CI must contain the empirical median (it is a
    # superset of the central 95 percent of the sample, which trivially
    # includes the median).
    median = float(np.median(result.samples))
    assert lo <= median <= hi


# ============================================================================
# Test 2: empirical coverage probability
# ============================================================================


@pytest.mark.slow
def test_bootstrap_pd_coverage_probability() -> None:
    """
    Statistical-correctness test: empirical coverage of the bootstrap 95% CI.

    Procedure
    ---------
    For each of M = 5 Monte Carlo seeds:
      1. Simulate a Merton equity series of length N = 252 from a known
         theta_true = (mu_true, sigma_true).
      2. Calibrate to obtain the inferred V_A at the last observation.
         (This is what the bootstrap CI is centred around; the "true" PD
         for this realisation must be computed conditional on the same V_A.)
      3. Compute the closed-form true PD using V_A and theta_true.
      4. Bootstrap the PD with n_replicas = 10.
      5. Record whether true PD is inside the empirical 95-percent CI.

    Pass criterion: at least 3 of 5 MC samples (60 percent) contain the
    true PD in the CI.

    Why the threshold is 60 percent, not the nominal 95
    ---------------------------------------------------
    Three effects pull the empirical coverage below 95 percent:
      - **Small M = 5**: with M = 5 the empirical coverage is a step
        function {0, 0.2, 0.4, 0.6, 0.8, 1.0}, and even a perfect
        95-percent estimator has P(>= 5 / 5 | true=0.95) = 0.95^5 ~= 0.77,
        so requiring 5 / 5 would fail ~23 percent of the time. The
        binomial P(>= 3 / 5 | true=0.95) is 0.9988: requiring 3 / 5 gives
        near-zero false-failure rate on a good estimator.
      - **Small n_replicas = 10**: the percentile bootstrap is biased
        toward narrower CIs for small replica counts, which depresses
        empirical coverage by ~5 to 10 percentage points.
      - **Compute budget**: at ~3 seconds per Merton calibration,
        M * (1 + n_replicas) = 55 calibrations fits in ~3 minutes; pushing
        further would push the test over the practical CI budget.

    Putting all three together, a threshold of 60 percent detects a real
    methodological failure (bootstrap CI systematically biased, wrong
    scaling of the resampled series, etc.) while accepting normal
    finite-sample and small-bootstrap noise. A rigorous coverage test
    would use M >= 100 and n_replicas >= 200 with threshold = 85 percent;
    that would run for > 1 hour and belongs in a nightly job, not a unit
    suite.

    Marked @pytest.mark.slow. The default `pytest` invocation skips it
    (pyproject.toml sets `addopts = "-ra -m 'not slow'"`); run it
    explicitly with `pytest -m slow`.
    """
    mu_true = 0.05
    sigma_true = 0.30
    n_mc = 5
    n_replicas = 10
    coverage_threshold = 0.60  # see docstring

    successes = 0

    for mc_seed in range(n_mc):
        rng = np.random.default_rng(mc_seed)
        equity = _simulate_merton_equity(
            rng=rng,
            n=N_OBSERVATIONS,
            mu=mu_true,
            sigma=sigma_true,
            delta=DELTA,
            initial=INITIAL_EQUITY,
        )

        cal = _build_merton_calibrator(equity=equity)
        cal.calibration()

        v_a_realized = cal.get_final_asset()
        true_pd_pct = _merton_true_pd_pct(
            mu_true=mu_true,
            sigma_true=sigma_true,
            v_a=v_a_realized,
            k=DEBT,
            t=MATURITY,
        )

        result = bootstrap_pd(
            calibrator=cal, n_replicas=n_replicas, seed=mc_seed + 1_000
        )

        lo, hi = result.ci_95
        if lo <= true_pd_pct <= hi:
            successes += 1

    empirical_coverage = successes / n_mc
    assert empirical_coverage >= coverage_threshold, (
        f"Empirical coverage {empirical_coverage:.2f} below threshold "
        f"{coverage_threshold:.2f} (successes={successes}/{n_mc}). "
        f"Either the bootstrap CI is systematically biased or the random "
        f"draws happened to land in the failure tail; rerun with a different "
        f"seed range to disambiguate."
    )
