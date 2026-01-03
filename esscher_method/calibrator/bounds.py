# esscher_method/calibrator/bounds.py
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def bounds_penalty(
    parameter_vector: Sequence[float],
    bounds: Sequence[Tuple[float, float]],
    *,
    margin: float = 0.02,
    outside_weight: float = 50.0,
    normalize: bool = True,
) -> float:
    """
    Compute a smooth penalty that discourages solutions close to bounds and strongly penalizes values outside bounds.

    The penalty is designed to be small relative to the main objective and to never return NaN/inf.
    """
    if bounds is None or len(bounds) == 0:
        return 0.0

    values = np.asarray(parameter_vector, dtype=float).reshape(-1)
    bnds = list(bounds)

    num_params = min(int(values.size), int(len(bnds)))
    penalty = 0.0

    penalty += _penalty_for_dimension_mismatch(values=values, bounds=bnds, outside_weight=outside_weight)

    for v, lo, hi in _iter_parameter_bounds(values=values, bounds=bnds):
        if not _is_finite_valid_interval(param_value=v, lower_bound=lo, upper_bound=hi):
            penalty += float(outside_weight)
            continue

        outside = _penalty_if_outside_bounds(
            param_value=v,
            lower_bound=lo,
            upper_bound=hi,
            outside_weight=outside_weight,
        )
        if outside > 0.0:
            penalty += float(outside)
            continue

        penalty += _penalty_if_near_bounds(
            param_value=v,
            lower_bound=lo,
            upper_bound=hi,
            margin=margin,
        )

    if normalize:
        penalty = _normalize_penalty(penalty_sum=penalty, num_params=num_params)

    if not np.isfinite(penalty):
        return float(outside_weight)

    return float(penalty)


def _iter_parameter_bounds(
    *,
    values: np.ndarray,
    bounds: List[Tuple[float, float]],
) -> Iterable[Tuple[float, float, float]]:
    """
    Yield (value, lower, upper) for each dimension up to the minimum length.
    """
    count = min(int(values.size), int(len(bounds)))
    for i in range(count):
        lo, hi = bounds[i]
        yield float(values[i]), float(lo), float(hi)


def _penalty_for_dimension_mismatch(
    *,
    values: np.ndarray,
    bounds: List[Tuple[float, float]],
    outside_weight: float,
) -> float:
    """
    Penalize inconsistent shapes to guard against configuration errors.
    """
    if int(values.size) == int(len(bounds)):
        return 0.0
    mismatch = abs(int(values.size) - int(len(bounds)))
    return float(outside_weight * (mismatch + 1.0))


def _is_finite_valid_interval(
    *,
    param_value: float,
    lower_bound: float,
    upper_bound: float,
) -> bool:
    """
    Return True if inputs are finite and the interval is strictly increasing.
    """
    if not (np.isfinite(param_value) and np.isfinite(lower_bound) and np.isfinite(upper_bound)):
        return False
    return bool(float(upper_bound) > float(lower_bound))


def _penalty_if_outside_bounds(
    *,
    param_value: float,
    lower_bound: float,
    upper_bound: float,
    outside_weight: float,
) -> float:
    """
    Apply a quadratic penalty when a value is outside bounds.
    """
    lo = float(lower_bound)
    hi = float(upper_bound)
    if not (hi > lo):
        return 0.0

    span = hi - lo
    x = float(param_value)

    if x < lo:
        ratio = (lo - x) / span
        return float(outside_weight * (1.0 + ratio) ** 2)

    if x > hi:
        ratio = (x - hi) / span
        return float(outside_weight * (1.0 + ratio) ** 2)

    return 0.0


def _penalty_if_near_bounds(
    *,
    param_value: float,
    lower_bound: float,
    upper_bound: float,
    margin: float,
) -> float:
    """
    Apply a smooth penalty when a value is within a relative margin of either bound.
    """
    lo = float(lower_bound)
    hi = float(upper_bound)
    if not (hi > lo):
        return 0.0

    x = float(param_value)
    if x < lo or x > hi:
        return 0.0

    span = hi - lo
    rel_lo = (x - lo) / span
    rel_hi = (hi - x) / span
    nearest = min(rel_lo, rel_hi)

    m = float(margin)
    if nearest >= m:
        return 0.0

    scaled = (m - nearest) / m
    return float(scaled * scaled)


def _normalize_penalty(*, penalty_sum: float, num_params: int) -> float:
    """
    Normalize by the number of parameters to keep scale comparable across models.
    """
    return float(penalty_sum / max(1, int(num_params)))
