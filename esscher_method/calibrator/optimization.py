# esscher_method/calibrator/optimization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from esscher_method.model.model import Model
from .data_calibration import CalibrationConfig
from .moment_matching import MomentMatcher


@dataclass(frozen=True)
class LeastSquaresConfig:
    """
    Numerical settings for bounded nonlinear least squares.
    """
    xtol: float = 1e-12
    ftol: float = 1e-12
    gtol: float = 1e-12
    max_nfev: int = 10_000
    accept_sse_threshold: float = 10.0


class ParameterOptimizer:
    """
    Estimate physical parameters via bounded least squares on moment residuals,
    with optional global optimization fallback.
    """

    def __init__(
        self,
        model: Model,
        config: CalibrationConfig,
        matcher: MomentMatcher,
        lsq_config: Optional[LeastSquaresConfig] = None,
    ) -> None:
        """
        Initialize the optimizer used by Calibrator during physical-parameter calibration steps
        (moment matching in STEP 1 on equity log-returns and STEP 4 on inferred asset log-returns).
        """

        if model is None:
            raise ValueError("model must not be None.")
        if config is None:
            raise ValueError("config must not be None.")
        if matcher is None:
            raise ValueError("matcher must not be None.")

        self.model = model
        self.config = config
        self.matcher = matcher
        self.lsq_config = lsq_config or LeastSquaresConfig()

    def estimate_parameters(self, initial_guess: np.ndarray) -> Dict[str, float]:
        """
        Estimate physical (P-measure) model parameters for Calibrator's moment-matching stages
        (STEP 1 and STEP 4), returning a name-to-value mapping aligned with model.parameter_names.
        """

        if self.model.parameters is None:
            raise ValueError("model.parameters must not be None.")

        bounds = getattr(self.model, "bounds", None)
        if bounds is None or len(bounds) == 0:
            raise ValueError("model.bounds must be defined and non-empty.")

        names = list(getattr(self.model, "parameter_names", ()))
        if not names:
            names = list(self.model.parameters.keys())

        if len(bounds) != len(names):
            raise ValueError(
                "Inconsistent configuration: model.bounds length must match model.parameter_names length "
                f"(bounds={len(bounds)}, names={len(names)})."
            )

        x0 = np.asarray(initial_guess, dtype=float).reshape(-1)
        if x0.size != len(names):
            raise ValueError(
                "initial_guess size must match the number of calibrated parameters "
                f"(initial_guess={x0.size}, expected={len(names)})."
            )

        x_ls = self._run_bounded_least_squares(x0=x0, bounds=list(bounds))
        if self._least_squares_solution_is_acceptable(x_ls):
            return {name: float(val) for name, val in zip(names, x_ls)}

        x_opt = self._run_fallback_optimizers(x0=x0, bounds=list(bounds))
        return {name: float(val) for name, val in zip(names, x_opt)}


    def _least_squares_solution_is_acceptable(self, x: np.ndarray) -> bool:
        """
        Internal quality gate for the bounded least-squares candidate used in physical-parameter estimation;
        called by estimate_parameters to decide whether to accept least-squares or trigger fallbacks.
        """

        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0 or (not np.all(np.isfinite(x))):
            return False

        r = self.matcher.residual_vector(x)
        if r.size == 0 or (not np.all(np.isfinite(r))):
            return False

        sse = float(np.sum(r * r))
        if not np.isfinite(sse):
            return False

        return bool(sse < float(self.lsq_config.accept_sse_threshold))

    def _run_bounded_least_squares(
        self,
        *,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Primary solver for physical-parameter estimation in Calibrator's moment matching (STEP 1 / STEP 4),
        fitting parameters via bounded nonlinear least squares on MomentMatcher residuals.
        """

        x0 = np.asarray(x0, dtype=float).reshape(-1)

        lower = np.array([float(b[0]) for b in bounds], dtype=float)
        upper = np.array([float(b[1]) for b in bounds], dtype=float)

        x0 = np.clip(x0, lower, upper)

        res = optimize.least_squares(
            fun=self.matcher.residual_vector,
            x0=x0,
            bounds=(lower, upper),
            xtol=float(self.lsq_config.xtol),
            ftol=float(self.lsq_config.ftol),
            gtol=float(self.lsq_config.gtol),
            max_nfev=int(self.lsq_config.max_nfev),
        )

        if not bool(res.success):
            return np.asarray(res.x, dtype=float).reshape(-1)

        x = np.asarray(res.x, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            return np.clip(x0, lower, upper)

        return x

    def _run_fallback_optimizers(
        self,
        *,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Fallback optimization path for physical-parameter estimation when bounded least squares is not acceptable;
        called by estimate_parameters to run optional global search and bounded local refinement.
        """

        x0 = np.asarray(x0, dtype=float).reshape(-1)

        if bool(self.config.minimization_diff_evolution) and x0.size > 1:
            x_global = self._run_differential_evolution(bounds=bounds)
            return self._run_local_minimizer(x0=x_global, bounds=bounds)

        return self._run_local_minimizer(x0=x0, bounds=bounds)

    def _run_local_minimizer(
        self,
        *,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Bounded local refinement used within physical-parameter estimation (STEP 1 / STEP 4),
        either after global initialization or as a direct fallback; called by _run_fallback_optimizers.
        """

        x0 = np.asarray(x0, dtype=float).reshape(-1)

        res = optimize.minimize(
            fun=self.matcher.objective,
            x0=x0,
            bounds=bounds,
            method=str(self.config.local_minimizer_method),
            options={"maxiter": int(self.config.local_maxiter)},
        )

        x = np.asarray(res.x, dtype=float).reshape(-1)
        if x.size != x0.size:
            return x0

        if not np.all(np.isfinite(x)):
            return x0

        return x

    def _run_differential_evolution(self, *, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Optional global search used only as a fallback initializer for physical-parameter estimation
        in Calibrator (moment matching in STEP 1 and STEP 4).

        This method is invoked by _run_fallback_optimizers() only when:
        - config.minimization_diff_evolution is True (enables global search), and
        - the parameter dimension is greater than 1 (x0.size > 1), since DE is unnecessary for 1D problems.

        It is not used when:
        - config.minimization_diff_evolution is False (local-only calibration), or
        - the parameter vector has size 1 (local bounded minimization is used directly).

        When enabled, DE runs within the provided bounds using CalibrationConfig settings. 
        Its output is a robust starting point that is subsequently refined by
        _run_local_minimizer(), which performs the final bounded local optimization.
        """


        res = optimize.differential_evolution(
            self.matcher.objective,
            bounds=bounds,
            strategy=str(self.config.de_strategy),
            maxiter=int(self.config.de_maxiter),
            popsize=int(self.config.de_popsize),
            tol=float(self.config.de_tol),
            mutation=self.config.de_mutation,
            recombination=float(self.config.de_recombination),
            polish=bool(self.config.de_polish),
            init=str(self.config.de_init),
            updating=str(self.config.de_updating),
            seed=self.config.de_seed,
            workers=int(self.config.de_workers),
        )

        x = np.asarray(res.x, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            x = np.array([(lo + hi) * 0.5 for lo, hi in bounds], dtype=float)

        return x
