"""
Test utilities for the nightly recovery-study suite (P3.T4).

This module collects the heavier helper code so that the test file
itself can stay focused on the test logic. Nothing in this module is
collected as a test by pytest (the filename starts with an underscore).

================================================================================
Contents
================================================================================

1. Levy DGP simulation
   - simulate_log_returns: draw N iid daily log-return increments from the
     canonical Levy representation of Merton / BG / VG.

2. Model construction
   - fresh_model_with_theta: build a fresh model instance with the given
     parameter dictionary applied. Used both for tolerance derivation
     (needs theta_true) and for forward Lewis pricing (needs RN-state model).

3. Sample-cumulant computation
   - sample_cumulants_matching_keys: replicate the unbiased k-statistics
     used internally by MomentMatcher.update_sample_cumulants. Keeping the
     formulas in sync is load-bearing for the delta-method tolerance to
     match what the calibrator actually optimizes.

4. Theoretical tolerance derivation (Fisher / delta method)
   - cumulant_jacobian_at_theta: numerical Jacobian
     J = d(k_theoretical_daily) / d(theta) at theta_true.
   - sigma_cumulant_montecarlo: Monte-Carlo estimate of Cov(k_sample)
     under theta_true.
   - fisher_tolerance: top-level entry that produces a per-parameter
     absolute tolerance via Cov(theta_hat) = J^{-1} Sigma J^{-T}.

5. Forward equity construction
   - build_equity_from_asset: produce a coherent V_E series from a
     simulated V_A series by forward Lewis pricing under the RN measure
     derived from theta_true via the Esscher transform. This is the
     non-deep-ITM setup that exercises the full calibration pipeline.

================================================================================
Mathematical background (Fisher / delta-method tolerance)
================================================================================

The calibrator identifies theta by solving the moment-matching system:

    k_sample = k_theoretical(theta)

where k_sample is the vector of unbiased sample cumulants of the daily
log-returns and k_theoretical(theta) is the model's theoretical-cumulant
mapping.

A first-order Taylor expansion at theta_true gives:

    theta_hat - theta_true  =  J^{-1} (k_sample - k_true) + o_p(1/sqrt(N))

where J = d k_theoretical / d theta evaluated at theta_true (square when
the number of cumulants matched equals the number of parameters; this is
the case for all three models: 2x2 for Merton, 4x4 for BG, 3x3 for VG).

Asymptotic covariance:

    Cov(theta_hat)  =  J^{-1} Cov(k_sample) J^{-T}

Cov(k_sample) is estimated by Monte Carlo (simulating n_replicas
independent sample paths of length N from theta_true and computing
empirical covariance of the resulting k_sample vectors). The MC
approach avoids deriving the closed-form variance formulas of the
sample cumulants (which involve cumulants up to order 2 * max_order
and become unwieldy beyond order 2).

Per-parameter z-sigma tolerance:

    abs_tol[j]  =  z * sqrt(Cov(theta_hat)[j, j])

z = 4 in the default setup (4-sigma envelope; the rejection rate
P(|theta_hat - theta_true| > z * SE) is ~6e-5 under the asymptotic
Gaussian).

================================================================================
Caveats
================================================================================

The asymptotics target "moment matching on observed X" - i.e., the
calibrator fit on V_A log-returns if V_A were observed directly. The
actual calibrator infers V_A from V_E via Lewis inversion, which adds
a (small, asymptotically negligible) layer of estimation error. We
assume z = 4 gives sufficient headroom for this; if the nightly test
fails the tolerance derived here, investigate whether the discrepancy
is due to (a) finite-N higher-order terms in the delta-method
expansion, (b) Lewis inversion noise, or (c) a real identification
problem (e.g., BG cumulant-equivalent local minima). Do NOT relax z
without one of these explanations.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

import numpy as np
from scipy.stats import moment

from esscher_method.calibrator.esscher_solver import EsscherSolver
from esscher_method.calibrator.pricer import LewisEuropeanTargetPricer, LewisPricerConfig
from esscher_method.model.model import BilateralGamma, Merton, Model, VarianceGamma


# ============================================================================
# Model registry
# ============================================================================

MODEL_REGISTRY: Dict[str, Type[Model]] = {
    "Merton": Merton,
    "BilateralGamma": BilateralGamma,
    "VarianceGamma": VarianceGamma,
}


# ============================================================================
# 1. Levy DGP simulation
# ============================================================================


def simulate_log_returns(
    model_name: str,
    theta_true: Dict[str, float],
    rng: np.random.Generator,
    n: int,
    delta: float,
) -> np.ndarray:
    """
    Simulate n iid daily log-return increments from the named DGP.

    For Merton: Normal(mu * delta, sigma^2 * delta) increments.
    For BG / VG: difference of two independent Gamma random variables, in
    the canonical Levy representation X = G_P - G_M where G_P, G_M are
    Gamma with shape alpha_* * delta and scale 1 / lambda_*.

    Parameters
    ----------
    model_name : {"Merton", "BilateralGamma", "VarianceGamma"}
    theta_true : dict mapping parameter names to values (in the model's
        native parameterisation, NOT the conventional (sigma, theta, nu)
        form for VG).
    rng : numpy Generator.
    n : number of daily increments.
    delta : time step in years (typically 1/252).

    Returns
    -------
    np.ndarray of shape (n,) with iid daily log-returns.
    """
    if model_name == "Merton":
        mu = theta_true["mu"]
        sigma = theta_true["sigma"]
        return rng.normal(loc=mu * delta, scale=sigma * float(np.sqrt(delta)), size=n)

    if model_name == "BilateralGamma":
        gp = rng.gamma(
            shape=theta_true["alpha_P"] * delta,
            scale=1.0 / theta_true["lambda_P"],
            size=n,
        )
        gm = rng.gamma(
            shape=theta_true["alpha_M"] * delta,
            scale=1.0 / theta_true["lambda_M"],
            size=n,
        )
        return gp - gm

    if model_name == "VarianceGamma":
        a = theta_true["alpha"]
        gp = rng.gamma(shape=a * delta, scale=1.0 / theta_true["lambda_P"], size=n)
        gm = rng.gamma(shape=a * delta, scale=1.0 / theta_true["lambda_M"], size=n)
        return gp - gm

    raise ValueError(f"Unknown model: {model_name!r}")


# ============================================================================
# 2. Model construction
# ============================================================================


def fresh_model_with_theta(
    model_name: str,
    delta: float,
    theta: Dict[str, float],
) -> Model:
    """
    Build a fresh model instance with parameters set to theta.

    Used (a) by the calibrator helpers for jacobian computation and
    (b) by build_equity_from_asset to set up the RN-state pricer.
    """
    cls = MODEL_REGISTRY[model_name]
    m = cls(delta=delta)
    param_vec = np.array([theta[k] for k in m.parameter_names], dtype=float)
    m.parameters_update(param_vec)
    return m


# ============================================================================
# 3. Sample cumulants matching MomentMatcher
# ============================================================================


def sample_cumulants_matching_keys(observations: np.ndarray, keys: Tuple[str, ...]) -> np.ndarray:
    """
    Compute the unbiased sample cumulants matching the keys returned by
    Model.theoretical_cumulants().

    The formulas mirror MomentMatcher.update_sample_cumulants exactly so
    that the delta-method tolerance reflects what the calibrator actually
    minimises. If MomentMatcher ever changes its sample-cumulant
    formulas, this helper must change in lockstep.

    Supported keys: "mean", "variance", "skewness", "fourth_cumulant".
    """
    x = np.asarray(observations, dtype=float).reshape(-1)
    n = int(x.size)
    if n < 4:
        raise ValueError(f"Need at least 4 observations for k-statistics, got {n}.")
    out = []
    for key in keys:
        if key == "mean":
            out.append(float(np.mean(x)))
        elif key == "variance":
            out.append(float(np.var(x, ddof=1)))
        elif key == "skewness":
            m3 = float(moment(x, moment=3))
            out.append(float((n * n) / ((n - 1) * (n - 2)) * m3))
        elif key == "fourth_cumulant":
            m2 = float(moment(x, moment=2))
            m4 = float(moment(x, moment=4))
            out.append(
                float(
                    (n * n) / ((n - 1) * (n - 2) * (n - 3))
                    * ((n + 1) * m4 - 3.0 * (n - 1) * (m2 * m2))
                )
            )
        else:
            raise ValueError(f"Unknown cumulant key: {key!r}")
    return np.asarray(out, dtype=float)


# ============================================================================
# 4. Theoretical tolerance derivation
# ============================================================================


def cumulant_jacobian_at_theta(
    model_name: str,
    delta: float,
    theta_true: Dict[str, float],
    h_rel: float = 1e-5,
) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """
    Numerical Jacobian J = d(k_theoretical_daily) / d(theta) at theta_true.

    Uses centered finite differences with relative step h_rel * |theta_j|
    (or h_rel if |theta_j| < 1). Returns J of shape
    (n_cumulants, n_params), along with the param_names and cumulant_keys
    in the same order used for the rows / columns of J.

    Each finite-difference probe instantiates a fresh model to avoid
    contaminating state between calls (the model classes are stateful
    via their `parameters` dict).
    """
    base = fresh_model_with_theta(model_name, delta, theta_true)
    cumulant_keys = tuple(base.theoretical_cumulants().keys())
    param_names = tuple(base.parameter_names)
    n_params = len(param_names)
    n_cumulants = len(cumulant_keys)

    theta_vec = np.array([theta_true[k] for k in param_names], dtype=float)
    J = np.zeros((n_cumulants, n_params))

    for j in range(n_params):
        h = h_rel * max(abs(theta_vec[j]), 1.0)

        theta_plus = dict(theta_true)
        theta_plus[param_names[j]] = float(theta_vec[j] + h)
        theta_minus = dict(theta_true)
        theta_minus[param_names[j]] = float(theta_vec[j] - h)

        m_plus = fresh_model_with_theta(model_name, delta, theta_plus)
        m_minus = fresh_model_with_theta(model_name, delta, theta_minus)

        k_plus = np.array(list(m_plus.theoretical_cumulants().values()), dtype=float)
        k_minus = np.array(list(m_minus.theoretical_cumulants().values()), dtype=float)

        J[:, j] = (k_plus - k_minus) / (2.0 * h)

    return J, param_names, cumulant_keys


def sigma_cumulant_montecarlo(
    model_name: str,
    theta_true: Dict[str, float],
    delta: float,
    n_obs: int,
    cumulant_keys: Tuple[str, ...],
    n_replicas: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Empirical Cov(k_sample) where k_sample is the vector of sample
    cumulants matching cumulant_keys, computed from n_obs iid simulated
    daily log-returns from theta_true.

    Note on scaling: the returned covariance is Cov(k_sample) (i.e., it
    already includes the 1/N scaling implicit in finite-sample estimators).
    Do NOT divide by N when propagating through the Jacobian.

    Cost: n_replicas independent simulations of length n_obs. Cheap
    (only random draws + k-statistics, no calibration).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_k = len(cumulant_keys)
    samples = np.zeros((n_replicas, n_k))
    for i in range(n_replicas):
        x = simulate_log_returns(model_name, theta_true, rng, n_obs, delta)
        samples[i, :] = sample_cumulants_matching_keys(x, cumulant_keys)

    return np.cov(samples, rowvar=False)


def fisher_tolerance(
    model_name: str,
    theta_true: Dict[str, float],
    delta: float,
    n_obs: int,
    z: float = 4.0,
    n_mc_for_sigma: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Per-parameter absolute tolerance via the delta-method:

        abs_tol[j] = z * sqrt(Cov(theta_hat)[j, j])
        Cov(theta_hat) = J^{-1} * Cov(k_sample) * J^{-T}
        J             = d k_theoretical_daily / d theta  (at theta_true)
        Cov(k_sample) = empirical from n_mc_for_sigma simulations of n_obs

    Parameters
    ----------
    model_name : {"Merton", "BilateralGamma", "VarianceGamma"}
    theta_true : dict, true parameters used as the linearisation point.
    delta : daily time step (years).
    n_obs : sample size of the recovery test (number of daily increments).
    z : z-sigma envelope (default 4.0). Rejection rate at z=4 is ~6e-5
        under the asymptotic Gaussian.
    n_mc_for_sigma : number of MC paths used to estimate Cov(k_sample).
        500 gives empirical SE on each variance entry of order ~6.5%,
        which is fine for a z=4 tolerance.
    rng : optional Generator for reproducibility of the MC step.

    Returns
    -------
    dict mapping parameter names to absolute tolerances.

    Notes
    -----
    All three currently supported models are exactly identified
    (n_cumulants == n_params), so J is square and invertible at any
    interior theta. The implementation uses np.linalg.inv; if a future
    model adds an over-identified setup (more cumulants than params),
    the fallback below switches to (J^T J)^{-1} J^T weighting.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    J, param_names, cumulant_keys = cumulant_jacobian_at_theta(
        model_name=model_name, delta=delta, theta_true=theta_true
    )

    sigma_k = sigma_cumulant_montecarlo(
        model_name=model_name,
        theta_true=theta_true,
        delta=delta,
        n_obs=n_obs,
        cumulant_keys=cumulant_keys,
        n_replicas=n_mc_for_sigma,
        rng=rng,
    )

    n_cumulants, n_params = J.shape
    if n_cumulants == n_params:
        J_inv = np.linalg.inv(J)
        cov_theta = J_inv @ sigma_k @ J_inv.T
    else:
        # Over-identified moment matching: pseudoinverse weighting.
        # Not used by current models; kept for safety on future extensions.
        JTJ_inv = np.linalg.inv(J.T @ J)
        H = JTJ_inv @ J.T
        cov_theta = H @ sigma_k @ H.T

    se = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
    return {name: float(z * se[i]) for i, name in enumerate(param_names)}


# ============================================================================
# 5. Forward equity construction
# ============================================================================


def build_equity_from_asset(
    asset_values: np.ndarray,
    K: float,
    risk_free_rate: float,
    maturity: float,
    delta: float,
    theta_true: Dict[str, float],
    model_name: str,
    pricer_config: Optional[LewisPricerConfig] = None,
) -> np.ndarray:
    """
    Compute V_E[idx] = call_price(V_A[idx], K, ttm(idx), r) under the RN
    measure derived from theta_true via the Esscher transform.

    Uses retro-extrapolated ttm matching the calibrator's convention
    (see asset_inference.daily_asset_inversion docstring):

        ttm(idx) = maturity + (N - idx - 1) * delta

    so the LAST observation (idx = N - 1) prices a call with the nominal
    maturity, and earlier observations price longer-dated calls.

    This is the non-deep-ITM coherent setup that exercises the full
    calibration pipeline including the Lewis inversion step.

    Parameters
    ----------
    asset_values : np.ndarray of shape (N,), simulated V_A series
        consistent with theta_true under the physical measure (i.e.,
        cumulative exponentiation of simulate_log_returns output).
    K : strike (debt level).
    risk_free_rate : per-unit-time continuously compounded rate.
    maturity : nominal maturity at idx = N - 1.
    delta : daily time step (years).
    theta_true : true physical parameters used to derive the RN measure.
    model_name : {"Merton", "BilateralGamma", "VarianceGamma"}.
    pricer_config : optional Lewis pricer config; default has
        integration_upper_bound = 1024 and quad_limit = 500.

    Returns
    -------
    np.ndarray of shape (N,) with forward-priced call values.

    Implementation
    --------------
    1. Instantiate a fresh model in physical state with theta_true.
    2. Solve the Esscher equation K_X(p_star + 1) - K_X(p_star) = r * delta
       for p_star.
    3. Update the model's risk_neutral_parameters via
       Model.risk_neutral_parameters_update(p_star).
    4. Call LewisEuropeanTargetPricer.price(S0) for each asset value at
       the corresponding ttm. The pricer reads model.chf(..., risk_neutral=True)
       internally, which uses the RN parameters set in step 3.
    """
    pricer_config = pricer_config or LewisPricerConfig()

    physical_model = fresh_model_with_theta(model_name, delta, theta_true)

    solver = EsscherSolver(model=physical_model, risk_free_rate=risk_free_rate, delta=delta)
    p_star = solver.solve()

    physical_model.risk_neutral_parameters = physical_model.risk_neutral_parameters_update(
        p_star=p_star
    )

    pricer = LewisEuropeanTargetPricer(
        model=physical_model,
        K=float(K),
        risk_free_rate=float(risk_free_rate),
        config=pricer_config,
    )

    av = np.asarray(asset_values, dtype=float).reshape(-1)
    n = int(av.size)
    equity = np.zeros(n, dtype=float)
    for idx in range(n):
        ttm = float(maturity) + float(n - idx - 1) * float(delta)
        pricer.T = ttm
        equity[idx] = float(pricer.price(S0=float(av[idx])))
    return equity
