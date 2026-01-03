# esscher_method/calibrator/asset_inference.py
from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

from esscher_method.model.model import Model
from .data_calibration import AssetInversionConfig, CalibrationConfig, CalibrationData
from .pricer import LewisEuropeanTargetPricer, LewisPricerConfig


class SquaredResidualObjective:
    """
    Callable objective used as a robust fallback when root bracketing fails.
    """

    def __init__(self, pricer: LewisEuropeanTargetPricer) -> None:
        self._pricer = pricer

    def __call__(self, x: np.ndarray) -> float:
        s = float(np.asarray(x, dtype=float).reshape(-1)[0])
        try:
            r = float(self._pricer.price_residual(S0=s))
        except Exception:
            return float("inf")
        if not np.isfinite(r):
            return float("inf")
        return float(r * r)


class AssetInverter:
    """
    Invert an observed equity value into an implied asset value within bounds.
    """

    def __init__(self, config: Optional[AssetInversionConfig] = None) -> None:
        self._cfg = config or AssetInversionConfig()

    def invert(
        self,
        *,
        pricer: LewisEuropeanTargetPricer,
        target_price: float,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        cfg = self._cfg

        pricer.target_price = float(target_price)

        lo = max(float(lower_bound), float(cfg.min_positive))
        hi_cap = max(float(upper_bound), lo)

        if hi_cap <= lo:
            return float(lo)

        r_lo = self._safe_residual(pricer, lo)
        if np.isfinite(r_lo) and abs(float(r_lo)) <= float(cfg.residual_tol):
            return float(lo)

        lo_b, hi_b, ok = self._build_bracket(pricer=pricer, lo=lo, hi_cap=hi_cap)

        if ok:
            try:
                sol = optimize.root_scalar(
                    pricer.price_residual,
                    bracket=(float(lo_b), float(hi_b)),
                    method="brentq",
                    xtol=float(cfg.xtol),
                    rtol=float(cfg.rtol),
                    maxiter=int(cfg.maxiter),
                )
                if bool(sol.converged) and np.isfinite(sol.root):
                    return float(sol.root)
            except Exception:
                pass

        return float(self._fallback_minimize(pricer=pricer, lo=lo, hi=hi_cap))

    def _safe_residual(self, pricer: LewisEuropeanTargetPricer, s: float) -> float:
        try:
            return float(pricer.price_residual(S0=float(s)))
        except Exception:
            return float("nan")

    def _build_bracket(
        self,
        *,
        pricer: LewisEuropeanTargetPricer,
        lo: float,
        hi_cap: float,
    ) -> Tuple[float, float, bool]:
        cfg = self._cfg

        lo = float(lo)
        hi_cap = float(max(hi_cap, lo))

        r_lo = self._safe_residual(pricer, lo)
        if not np.isfinite(r_lo):
            return lo, lo, False

        hi = min(hi_cap, max(lo + 1.0, lo * 1.1))
        r_hi = self._safe_residual(pricer, hi)

        while np.isfinite(r_hi) and (float(r_hi) < 0.0) and (hi < hi_cap):
            hi = min(hi_cap, hi * float(cfg.bracket_expand))
            r_hi = self._safe_residual(pricer, hi)

        ok = np.isfinite(r_lo) and np.isfinite(r_hi) and (float(r_lo) <= 0.0) and (float(r_hi) >= 0.0)
        return lo, hi, bool(ok)

    def _fallback_minimize(self, *, pricer: LewisEuropeanTargetPricer, lo: float, hi: float) -> float:
        cfg = self._cfg

        obj = SquaredResidualObjective(pricer)
        x0 = min(float(hi), max(float(lo), float(lo + 1.0)))

        res = optimize.minimize(
            obj,
            x0=np.array([x0], dtype=float),
            bounds=[(float(lo), float(hi))],
            method="Powell",
            options={"maxiter": int(cfg.maxiter)},
        )
        return float(np.asarray(res.x, dtype=float).reshape(-1)[0])


class DailyAssetInferenceWorker:
    """
    Picklable worker for process-based parallelism.
    """

    def __init__(
        self,
        *,
        model: Model,
        equity_values: np.ndarray,
        debt: float,
        maturity: float,
        delta: float,
        risk_free_rate: float,
        pricer_config: LewisPricerConfig,
        inversion_config: AssetInversionConfig,
        asset_upper_bound: float,
    ) -> None:
        self.model = model
        self.equity_values = np.asarray(equity_values, dtype=float).reshape(-1)
        self.debt = float(debt)
        self.maturity = float(maturity)
        self.delta = float(delta)
        self.risk_free_rate = float(risk_free_rate)
        self.pricer_config = pricer_config
        self.inversion_config = inversion_config
        self.asset_upper_bound = float(asset_upper_bound)

        self._inverter: Optional[AssetInverter] = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_inverter"] = None
        return state

    def _get_inverter(self) -> AssetInverter:
        if self._inverter is None:
            self._inverter = AssetInverter(config=self.inversion_config)
        return self._inverter

    def __call__(self, day_index: int) -> float:
        idx = int(day_index)
        total_days = int(self.equity_values.size)

        pricer = LewisEuropeanTargetPricer(
            model=self.model,
            K=float(self.debt),
            risk_free_rate=float(self.risk_free_rate),
            config=self.pricer_config,
        )

        ttm = float(self.maturity) + (total_days - idx - 1) * float(self.delta)
        pricer.T = float(ttm)

        target = float(self.equity_values[idx])
        inverter = self._get_inverter()

        return float(
            inverter.invert(
                pricer=pricer,
                target_price=target,
                lower_bound=target,
                upper_bound=float(self.asset_upper_bound),
            )
        )


class AssetInferenceEngine:
    """
    Infer asset values from observed equity values by inverting a European call price map.
    """

    def __init__(
        self,
        *,
        model: Model,
        data: CalibrationData,
        config: CalibrationConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if model is None:
            raise ValueError("model must not be None.")
        if data is None:
            raise ValueError("data must not be None.")
        if config is None:
            raise ValueError("config must not be None.")

        self.model = model
        self.data = data
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        pricer_cfg = self.config.pricer_config
        self.pricer_config = pricer_cfg if isinstance(pricer_cfg, LewisPricerConfig) else LewisPricerConfig()

        self.inverter = AssetInverter(config=self.config.asset_inversion)

    def infer_asset_values(self, *, day_indices: Sequence[int]) -> List[float]:
        """
        Infer asset values for the given day indices using serial or parallel execution.
        """
        idxs = [int(i) for i in day_indices]

        if not bool(self.config.use_parallel):
            return self._infer_serial(day_indices=idxs)

        kind = self._resolve_executor_kind()
        try:
            if kind == "thread":
                return self._infer_threaded(day_indices=idxs)
            if kind == "process":
                return self._infer_process(day_indices=idxs)
            raise ValueError(f"Invalid executor_kind: {self.config.executor_kind}")
        except Exception as exc:
            self._log(f"Parallel asset inference failed ({exc}). Falling back to serial.")
            return self._infer_serial(day_indices=idxs)

    def daily_asset_inversion(self, day_index: int) -> float:
        """
        Infer the asset value for a single day index.
        """
        idx = int(day_index)
        total_days = int(self.data.equity_array().size)

        pricer = LewisEuropeanTargetPricer(
            model=self.model,
            K=float(self.data.debt),
            risk_free_rate=float(self.data.risk_free_rate),
            config=self.pricer_config,
        )

        ttm = float(self.data.maturity) + (total_days - idx - 1) * float(self.model.delta)
        pricer.T = float(ttm)

        equity_series = self.data.equity_values
        try:
            target = float(getattr(equity_series, "iloc")[idx])
        except Exception:
            target = float(np.asarray(equity_series, dtype=float).reshape(-1)[idx])

        return float(
            self.inverter.invert(
                pricer=pricer,
                target_price=target,
                lower_bound=target,
                upper_bound=float(self.config.asset_upper_bound),
            )
        )

    def _build_worker(self) -> DailyAssetInferenceWorker:
        return DailyAssetInferenceWorker(
            model=self.model,
            equity_values=self.data.equity_array(),
            debt=float(self.data.debt),
            maturity=float(self.data.maturity),
            delta=float(self.model.delta),
            risk_free_rate=float(self.data.risk_free_rate),
            pricer_config=self.pricer_config,
            inversion_config=self.config.asset_inversion,
            asset_upper_bound=float(self.config.asset_upper_bound),
        )

    def _infer_serial(self, *, day_indices: List[int]) -> List[float]:
        return [float(self.daily_asset_inversion(i)) for i in day_indices]

    def _infer_threaded(self, *, day_indices: List[int]) -> List[float]:
        factory = self.config.thread_executor_factory
        with factory(max_workers=self.config.max_workers) as ex:
            results = list(ex.map(self.daily_asset_inversion, day_indices))
        return [float(v) for v in results]

    def _infer_process(self, *, day_indices: List[int]) -> List[float]:
        worker = self._build_worker()
        factory = self.config.executor_factory
        chunksize = int(self.config.map_chunksize)

        with factory(max_workers=self.config.max_workers) as ex:
            results = list(ex.map(worker, day_indices, chunksize=chunksize))
        return [float(v) for v in results]

    def _resolve_executor_kind(self) -> str:
        kind = str(self.config.executor_kind).lower()
        if kind != "auto":
            return kind
        if os.name == "nt" and bool(self.config.prefer_threads_on_windows):
            return "thread"
        return "process"

    def _log(self, message: str) -> None:
        if int(getattr(self.config, "verbose", 0)) > 0:
            self.logger.info(message)
