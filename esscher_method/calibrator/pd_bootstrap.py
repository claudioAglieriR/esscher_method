# esscher_method/calibrator/pd_bootstrap.py
"""
Bootstrap of the structural-model default probability (PD).

================================================================================
Purpose
================================================================================

The calibrator produces a single point estimate of the PD by inferring asset
values from observed equity, fitting physical Levy parameters, and inverting
the physical CDF at the maturity horizon. This is a *point* estimate: it
gives no indication of the sampling uncertainty arising from the finite
length of the equity time series.

This module quantifies that uncertainty by stationary-bootstrap resampling
of the equity log-returns and re-running the full calibration pipeline on
each resampled series. The resulting empirical distribution of PD estimates
yields a usable interval (mean, std, 2.5 / 97.5 percent quantiles).

================================================================================
Why stationary bootstrap (Politis-Romano, 1994)?
================================================================================

Equity log-returns exhibit serial dependence (notably volatility clustering)
even when the marginal distribution is reasonably modelled by an iid Levy
increment. A naive iid resampling of log-returns destroys this dependence
and underestimates the variance of any statistic that is sensitive to it
(including the calibrated physical parameters and hence the PD).

The stationary bootstrap of Politis and Romano draws blocks of consecutive
observations, with each block length geometrically distributed with mean
1/p. Wrapping the source series circularly removes edge effects so that the
resulting resampled series is *stationary* (every position has the same
expected value under the bootstrap distribution). The choice of block-length
distribution makes the procedure robust to the unknown autocorrelation
structure without requiring its explicit estimation.

The default mean block length is sqrt(N), the standard rule of thumb that
balances bias (long blocks recover long-range structure) against variance
(short blocks reduce within-block correlation in the bootstrap estimate).

================================================================================
What the bootstrap CI does NOT capture
================================================================================

The PD bootstrap CI quantifies sampling uncertainty *conditional on the
model*. It does not capture:

  - Model misspecification (Merton vs BG vs VG: pick the wrong family and
    the CI is around the wrong central value).
  - Parameter-set non-identifiability (BG has known cumulant-equivalent
    local minima at finite N; see test_recovery_study.py docstring).
  - Stale-debt / capital-structure-change risk: the calibrator assumes
    static debt (see CalibrationData docstring).

These are documented limitations of the methodology, not of the bootstrap
implementation.

================================================================================
Implementation cost
================================================================================

Each bootstrap replica re-runs the *full* calibration pipeline on the
resampled equity series. With `n_replicas = 200` (the default) and a Merton
calibration that takes ~1 second per replica, a single bootstrap call takes
~3 minutes. For BG/VG calibrations with iterative asset inversion the cost
scales accordingly. Callers can tune `n_replicas` to fit their compute
budget; ~50 replicas already give a usable CI for a coverage probability
near 95 percent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from esscher_method.model.model import Model

from .calibrator import Calibrator
from .data_calibration import CalibrationData


@dataclass(frozen=True)
class BootstrapPDResult:
    """
    Aggregated results from a bootstrap of the calibrated PD.

    Attributes
    ----------
    pd_mean : float
        Mean of the successful bootstrap PD samples (in percent, same units
        as Calibrator.default_probability_computation()).
    pd_std : float
        Sample standard deviation (ddof = 1) of the successful bootstrap PDs.
        NaN if fewer than 2 successful replicas.
    ci_95 : (float, float)
        Empirical 2.5 / 97.5 percent quantiles of the successful bootstrap
        PDs. This is a *percentile* CI, not a t-based one: appropriate when
        the bootstrap distribution may be skewed (which the PD typically is
        when it is small).
    samples : np.ndarray of shape (n_replicas,)
        The individual successful bootstrap PD estimates. Length equals
        n_replicas (the count of *successful* replicas, not the requested
        count).
    n_replicas : int
        Number of bootstrap replicas that completed successfully.
    n_failed : int
        Number of bootstrap replicas that raised an exception or produced a
        non-finite PD. A small number of failures is expected (the
        resampled series can occasionally land in a regime where the
        optimizer struggles); a large fraction indicates a problem with the
        original calibrator setup.
    """

    pd_mean: float
    pd_std: float
    ci_95: Tuple[float, float]
    samples: np.ndarray
    n_replicas: int
    n_failed: int


def bootstrap_pd(
    calibrator: Calibrator,
    *,
    n_replicas: int = 200,
    seed: Optional[int] = None,
    block_mean_length: Optional[float] = None,
) -> BootstrapPDResult:
    """
    Bootstrap the structural-model PD by stationary resampling of equity log-returns.

    For each replica:
      1. Draw a stationary-bootstrap sample of the original equity
         log-returns (same length as the original series).
      2. Reconstruct a synthetic equity level series starting from the
         original initial equity value.
      3. Instantiate a fresh model of the same type as the calibrator's
         model (with the same delta, policy, integration and domain
         settings), then a fresh Calibrator with the same configuration on
         the resampled data.
      4. Run the calibration and record the PD.

    The aggregated result is returned as a BootstrapPDResult.

    Parameters
    ----------
    calibrator : Calibrator
        The reference calibrator. Must have already been initialized with
        valid data; it does not need to have been calibrated yet. The
        reference calibrator itself is *not* mutated by this function: each
        replica uses fresh model and Calibrator instances. The calibrator's
        `data`, `model.delta`, `model.policy`, and `config` are read and
        used as templates.
    n_replicas : int, default 200
        Number of bootstrap replicas. Smaller values reduce runtime
        linearly; ~50 replicas already give a usable CI at 95 percent level.
    seed : int or None, default None
        Seed for the bootstrap RNG (controls both block-start indices and
        block lengths). Set to a fixed value for reproducibility. Note
        that each replica's *Calibrator* uses its own internal randomness
        (e.g., differential evolution); reproducibility across runs of
        bootstrap_pd therefore also requires the Calibrator's de_seed to
        be set on the input `calibrator.config`.
    block_mean_length : float or None, default None
        Mean block length for the stationary bootstrap. When None, defaults
        to sqrt(N) where N is the number of log-returns (the standard
        Politis-Romano rule of thumb).

    Returns
    -------
    BootstrapPDResult
        Aggregated bootstrap statistics; see the dataclass docstring.

    Notes
    -----
    Failures within a replica (calibration raising, or PD non-finite) are
    silently dropped from the aggregate. The number of dropped replicas is
    reported in the result's `n_failed` field; callers should check this
    is small relative to `n_replicas`.
    """
    if n_replicas < 1:
        raise ValueError(f"n_replicas must be >= 1, got {n_replicas}.")

    original_equity = np.asarray(calibrator.data.equity_array(), dtype=float)
    if original_equity.size < 2:
        raise ValueError("Need at least 2 equity observations to bootstrap log-returns.")

    log_returns = np.diff(np.log(original_equity))
    n_returns = int(log_returns.size)

    if block_mean_length is None:
        # Politis-Romano rule of thumb: mean block length ~ sqrt(N).
        block_mean_length = float(np.sqrt(n_returns))
    if block_mean_length < 1.0:
        raise ValueError(f"block_mean_length must be >= 1, got {block_mean_length}.")
    p_geom = 1.0 / float(block_mean_length)

    rng = np.random.default_rng(seed)
    initial_equity = float(original_equity[0])

    successes: list[float] = []
    n_failed = 0

    for _ in range(int(n_replicas)):
        boot_returns = _stationary_bootstrap_sample(
            rng=rng, x=log_returns, n_out=n_returns, p=p_geom
        )
        boot_equity = _reconstruct_equity_series(
            log_returns=boot_returns, initial=initial_equity
        )

        boot_data = CalibrationData(
            equity_values=boot_equity,
            debt=float(calibrator.data.debt),
            maturity=float(calibrator.data.maturity),
            risk_free_rate=float(calibrator.data.risk_free_rate),
        )
        fresh_model = _clone_model(calibrator.model)

        try:
            fresh_cal = Calibrator(model=fresh_model, data=boot_data, config=calibrator.config)
            fresh_cal.calibration()
            pd_pct = float(fresh_cal.default_probability_computation())
            if not np.isfinite(pd_pct):
                n_failed += 1
                continue
            successes.append(pd_pct)
        except Exception:
            # The bootstrap is intentionally robust to individual replica
            # failures: a resampled series can occasionally land in a
            # parameter regime where the optimizer struggles or the asset
            # inversion fails. Such replicas are dropped from the aggregate;
            # callers see the count via n_failed.
            n_failed += 1
            continue

    samples = np.asarray(successes, dtype=float)
    n_ok = int(samples.size)

    pd_mean = float(np.mean(samples)) if n_ok >= 1 else float("nan")
    pd_std = float(np.std(samples, ddof=1)) if n_ok >= 2 else float("nan")
    if n_ok >= 1:
        ci_lo = float(np.quantile(samples, 0.025))
        ci_hi = float(np.quantile(samples, 0.975))
    else:
        ci_lo = float("nan")
        ci_hi = float("nan")

    return BootstrapPDResult(
        pd_mean=pd_mean,
        pd_std=pd_std,
        ci_95=(ci_lo, ci_hi),
        samples=samples,
        n_replicas=n_ok,
        n_failed=n_failed,
    )


# ============================================================================
# Internal helpers
# ============================================================================


def _stationary_bootstrap_sample(
    *,
    rng: np.random.Generator,
    x: np.ndarray,
    n_out: int,
    p: float,
) -> np.ndarray:
    """
    Draw a single stationary-bootstrap (Politis-Romano) sample.

    The sample is built by concatenating circular blocks of x. Each block
    starts at a uniformly random index in [0, len(x)), and has a
    Geometric(p)-distributed length (so mean block length is 1/p). The
    source series x is treated as circular: an index past the end wraps
    around to the beginning, which keeps the bootstrap distribution
    stationary (every position is sampled from the same marginal).

    Vectorized form: pre-draw all block starts and lengths to avoid
    per-step Python overhead on long series.
    """
    n_src = int(x.shape[0])
    if n_src < 1:
        raise ValueError("Source array is empty; cannot bootstrap.")
    if n_out < 1:
        raise ValueError("Output length must be >= 1.")
    if not (0.0 < float(p) <= 1.0):
        raise ValueError(f"Geometric parameter p must be in (0, 1], got {p}.")

    # Worst case the loop draws n_out blocks of length 1; pre-allocate a
    # generous upper bound, then trim. This avoids a slow Python while-loop.
    # In expectation we draw n_out * p blocks; 2 * n_out * p + a small
    # safety margin is enough with overwhelming probability.
    block_pool = max(int(n_out * float(p) * 2 + 16), 8)

    out = np.empty(int(n_out), dtype=x.dtype)
    filled = 0
    while filled < n_out:
        block_starts = rng.integers(low=0, high=n_src, size=block_pool)
        block_lengths = rng.geometric(p=float(p), size=block_pool)
        for start, length in zip(block_starts, block_lengths):
            remaining = n_out - filled
            take = int(min(int(length), remaining))
            if take <= 0:
                continue
            # Wrap-around via modular indexing.
            indices = (int(start) + np.arange(take, dtype=np.int64)) % n_src
            out[filled : filled + take] = x[indices]
            filled += take
            if filled >= n_out:
                break
        # If the pool was exhausted without filling n_out (extremely
        # unlikely with the chosen pool size), the outer while loop refills.

    return out


def _reconstruct_equity_series(*, log_returns: np.ndarray, initial: float) -> np.ndarray:
    """
    Build an equity-level series from initial value and a sequence of log-returns.

    Returns an array of shape (len(log_returns) + 1,) so that taking
    np.diff(np.log(.)) of the result recovers `log_returns` exactly.
    """
    log_levels = float(np.log(initial)) + np.concatenate(
        ([0.0], np.cumsum(np.asarray(log_returns, dtype=float)))
    )
    return np.exp(log_levels)


def _clone_model(model: Model) -> Model:
    """
    Create a fresh model of the same concrete type as `model`.

    Preserves delta, policy, integration, and domain configs (so that any
    user-customized numerical settings carry over to the bootstrap
    replicas). Parameters are intentionally left at the policy defaults:
    each replica must calibrate from a generic starting point, otherwise
    the bootstrap would inherit the original fit's optimum and produce an
    artificially tight CI.
    """
    cls = type(model)
    return cls(
        delta=float(model.delta),
        policy=getattr(model, "policy", None),
        integration=getattr(model, "integration", None),
        domain=getattr(model, "domain", None),
    )
