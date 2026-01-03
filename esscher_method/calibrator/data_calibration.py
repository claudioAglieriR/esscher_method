# esscher_method/calibrator/data_calibration.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

Number = Union[int, float, np.number]


@dataclass(frozen=True)
class CalibrationData:
    """
    Immutable container for calibration inputs.

    Attributes:
        equity_values: Time series of observed equity values.
        debt: Debt level used as strike in the structural model.
        maturity: Horizon in years used for pricing and default probability.
        risk_free_rate: Continuously-compounded risk-free rate.
    """

    equity_values: Sequence[Number]
    debt: float
    maturity: float
    risk_free_rate: float = 0.0

    @property
    def num_observations(self) -> int:
        return int(len(self.equity_values))

    def equity_array(self) -> np.ndarray:
        return np.asarray(self.equity_values, dtype=float).reshape(-1)


@dataclass(frozen=True)
class AssetInversionConfig:
    """
    Numerical configuration for inverting an equity price into an implied asset value.

    Primary method: bracketed brentq root finding on the pricing residual.
    Fallback: bounded minimization of squared residual.
    """

    xtol: float = 1e-8
    rtol: float = 1e-8
    maxiter: int = 200
    bracket_expand: float = 2.0
    residual_tol: float = 1e-8
    min_positive: float = 1e-12


@dataclass(frozen=True)
class CalibrationConfig:
    """
    Numerical and algorithmic configuration for the calibration.

    The calibrator performs bounded least-squares moment matching by default, with
    optional global optimization fallback. Asset inference can run serially or in
    parallel using threads or processes.
    """

    minimization_diff_evolution: bool = True
    tolerance: float = 1e-7
    max_iterations: int = 20
    verbose: int = 1

    use_parallel: bool = True
    executor_kind: str = "auto"  # "auto", "thread", "process"
    prefer_threads_on_windows: bool = True
    max_workers: Optional[int] = None
    map_chunksize: int = 1

    executor_factory: Callable[..., ProcessPoolExecutor] = ProcessPoolExecutor
    thread_executor_factory: Callable[..., ThreadPoolExecutor] = ThreadPoolExecutor

    de_strategy: str = "best1bin"
    de_maxiter: int = 500
    de_tol: float = 1e-6
    de_mutation: Tuple[float, float] = (0.5, 1.5)
    de_recombination: float = 0.9
    de_polish: bool = True
    de_init: str = "latinhypercube"
    de_updating: str = "deferred"
    de_popsize: int = 20
    de_seed: Optional[int] = 12345
    de_workers: int = 1

    local_minimizer_method: str = "L-BFGS-B"
    local_maxiter: int = 2000

    cumulant_weights: Optional[Dict[str, float]] = None
    p_star_initial: float = 0.0

    asset_inversion: AssetInversionConfig = field(default_factory=AssetInversionConfig)
    asset_upper_bound: float = float(np.iinfo(np.int64).max)

    pricer_config: Optional[object] = None
