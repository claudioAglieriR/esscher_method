# esscher_method/calibrator/esscher_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import optimize

from esscher_method.model.model import Model


@dataclass(frozen=True)
class EsscherSolverConfig:
    """
    Numerical settings for solving the Esscher equation on a bounded interval.
    """
    grid_points: int = 81
    boundary_buffer_rel: float = 1e-6
    boundary_buffer_abs: float = 1e-8

    brentq_xtol: float = 1e-12
    brentq_rtol: float = 1e-10
    brentq_maxiter: int = 200

    lsq_tol: float = 1e-12
    lsq_max_nfev: int = 5000
    nan_penalty: float = 1e6


class EsscherSolver:
    """
    Solve the Esscher equation:
        K(p + 1) - K(p) = r * delta
    where K is the cumulant generating function under the physical measure.
    """

    def __init__(
        self,
        model: Model,
        *,
        risk_free_rate: float,
        delta: float,
        config: Optional[EsscherSolverConfig] = None,
    ) -> None:
        if model is None:
            raise ValueError("model must not be None.")
        self.model = model
        self.risk_free_rate = float(risk_free_rate)
        self.delta = float(delta)
        self.config = config or EsscherSolverConfig()

    def solve(self, *, p_initial: float = 0.0) -> float:
        """
        Solve for p_star within model-provided admissible bounds.
        """
        lower, upper = self.model.esscher_p_star_bounds()
        lower = float(lower)
        upper = float(upper)

        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise ValueError(f"Invalid Esscher bounds: lower={lower}, upper={upper}.")

        residual = self._residual_function()

        grid = self._build_grid(lower=lower, upper=upper, num_points=int(self.config.grid_points))
        exact_root, bracket = self._find_root_or_bracket(grid_points=grid, residual=residual)

        if exact_root is not None:
            return float(exact_root)

        if bracket is not None:
            root = self._solve_with_brentq(bracket=bracket, residual=residual)
            if root is not None:
                return float(root)

        return float(self._solve_with_bounded_least_squares(
            lower=lower,
            upper=upper,
            residual=residual,
            p_initial=float(p_initial),
        ))

    def _residual_function(self) -> Callable[[float], float]:
        # TODO : Discuss the usage of delta in rdt
        rdt = float(self.risk_free_rate) * float(self.delta)

        def residual(p: float) -> float:
            p = float(p)
            k_p1 = float(self.model.cumulant_generating_function(cgf_input=p + 1.0))
            k_p = float(self.model.cumulant_generating_function(cgf_input=p))
            if (not np.isfinite(k_p1)) or (not np.isfinite(k_p)):
                return float("nan")
            return float((k_p1 - k_p) - rdt)

        return residual

    def _build_grid(self, *, lower: float, upper: float, num_points: int) -> np.ndarray:
        lo = float(lower)
        hi = float(upper)
        if num_points < 3:
            num_points = 3

        span = hi - lo
        buf = max(float(self.config.boundary_buffer_abs), float(self.config.boundary_buffer_rel) * span)

        lo_safe = lo + buf
        hi_safe = hi - buf
        if not (lo_safe < hi_safe):
            lo_safe, hi_safe = lo, hi

        return np.linspace(lo_safe, hi_safe, num=int(num_points), dtype=float)

    def _find_root_or_bracket(
        self,
        *,
        grid_points: np.ndarray,
        residual: Callable[[float], float],
    ) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        prev_x: Optional[float] = None
        prev_y: Optional[float] = None

        for x in np.asarray(grid_points, dtype=float).reshape(-1):
            try:
                y = float(residual(float(x)))
            except Exception:
                continue

            if not np.isfinite(y):
                continue

            if y == 0.0:
                return float(x), None

            if prev_y is not None and np.isfinite(prev_y) and (prev_y * y < 0.0):
                return None, (float(prev_x), float(x))

            prev_x = float(x)
            prev_y = float(y)

        return None, None

    def _solve_with_brentq(
        self,
        *,
        bracket: Tuple[float, float],
        residual: Callable[[float], float],
    ) -> Optional[float]:
        a, b = float(bracket[0]), float(bracket[1])
        if not (np.isfinite(a) and np.isfinite(b) and a < b):
            return None

        try:
            sol = optimize.root_scalar(
                residual,
                bracket=(a, b),
                method="brentq",
                xtol=float(self.config.brentq_xtol),
                rtol=float(self.config.brentq_rtol),
                maxiter=int(self.config.brentq_maxiter),
            )
        except Exception:
            return None

        if bool(sol.converged) and np.isfinite(sol.root):
            return float(sol.root)
        return None

    def _solve_with_bounded_least_squares(
        self,
        *,
        lower: float,
        upper: float,
        residual: Callable[[float], float],
        p_initial: float,
    ) -> float:
        lo = float(lower)
        hi = float(upper)
        x0 = float(np.clip(float(p_initial), lo, hi))

        def safe_fun(x: np.ndarray) -> np.ndarray:
            p = float(np.asarray(x, dtype=float).reshape(-1)[0])
            val = float(residual(p))
            if not np.isfinite(val):
                val = float(self.config.nan_penalty)
            return np.array([val], dtype=float)

        sol = optimize.least_squares(
            fun=safe_fun,
            x0=np.array([x0], dtype=float),
            bounds=(np.array([lo], dtype=float), np.array([hi], dtype=float)),
            xtol=float(self.config.lsq_tol),
            ftol=float(self.config.lsq_tol),
            gtol=float(self.config.lsq_tol),
            max_nfev=int(self.config.lsq_max_nfev),
        )

        if (not bool(sol.success)) or (not np.isfinite(sol.x[0])):
            raise ValueError(f"Esscher bounded solver failed: {sol.message}")

        return float(sol.x[0])
