# esscher_method/calibrator/calibrator.py
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from esscher_method.model.model import Model
from .data_calibration import CalibrationConfig, CalibrationData
from .esscher_solver import EsscherSolver
from .moment_matching import MomentMatcher
from .optimization import ParameterOptimizer
from .pricer import LewisEuropeanTargetPricer, LewisPricerConfig
from .asset_inference import AssetInferenceEngine


class Calibrator:
    """
    Iterative calibration for structural Levy models using an Esscher transform.

    Workflow:
      1) Fit physical parameters on equity log-returns.
      2) Solve Esscher equation for p_star and update risk-neutral parameters.
      3) Infer daily asset values by inverting Lewis pricing.
      4) Fit physical parameters on inferred asset log-returns.
      5) Repeat until convergence or max_iterations is reached.
    """

    def __init__(self, model: Model, data: CalibrationData, config: Optional[CalibrationConfig] = None) -> None:
        if model is None:
            raise ValueError("model must be provided and cannot be None.")
        if data is None:
            raise ValueError("data must be provided and cannot be None.")

        self.model: Model = model
        self.data: CalibrationData = data
        self.config: CalibrationConfig = config or CalibrationConfig()

        equity_values = self.data.equity_array()
        if equity_values.size < 252:
            raise ValueError("equity_values must contain at least 252 observations.") #at least 1 year for calibration
        if float(self.data.debt) <= 0.0:
            raise ValueError("debt must be positive.")
        if float(self.data.maturity) <= 0.0:
            raise ValueError("maturity must be positive.")
        if float(self.model.delta) <= 0.0:
            raise ValueError("model.delta must be positive.")
        if int(self.config.max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1.")
        if float(self.config.tolerance) <= 0.0:
            raise ValueError("tolerance must be positive.")
        if int(self.config.map_chunksize) < 1:
            raise ValueError("map_chunksize must be >= 1.")

        kind = str(self.config.executor_kind).lower()
        if kind not in {"auto", "thread", "process"}:
            raise ValueError("executor_kind must be one of: auto, thread, process.")

        self.logger = logging.getLogger(self.__class__.__name__)
        self._configure_logger()

        pricer_cfg = self.config.pricer_config or LewisPricerConfig()
        self.pricer = LewisEuropeanTargetPricer(
            model=self.model,
            K=float(self.data.debt),
            risk_free_rate=float(self.data.risk_free_rate),
            config=pricer_cfg,
        )

        self.moment_matcher = MomentMatcher(model=self.model, config=self.config)
        self.optimizer = ParameterOptimizer(model=self.model, config=self.config, matcher=self.moment_matcher)
        self.esscher_solver = EsscherSolver(
            model=self.model,
            risk_free_rate=float(self.data.risk_free_rate),
            delta=float(self.model.delta),
        )
        self.asset_inference = AssetInferenceEngine(model=self.model, data=self.data, config=self.config, logger=self.logger)

        self.sample_cumulants: Dict[str, float] = {}
        self.physical_history: List[Dict[str, float]] = []
        self.risk_neutral_history: List[Dict[str, float]] = []

        self.iterations_number: int = 0
        self.p_star: float = float(self.config.p_star_initial)

        self.asset_values: List[float] = []
        self.distance_to_default: float = float("nan")
        self.final_asset_residual: float = float("nan")

    @property
    def equity_values_list(self) -> Sequence[float]:
        return self.data.equity_values

    @property
    def debt(self) -> float:
        return float(self.data.debt)

    @property
    def maturity(self) -> float:
        return float(self.data.maturity)

    @property
    def days_number(self) -> int:
        return int(self.data.equity_array().size)

    def _configure_logger(self) -> None:
        """
        Attach a StreamHandler to stdout if no handler is configured.
        """
        if self.logger.handlers:
            return

        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO if int(self.config.verbose) > 0 else logging.WARNING)

    def _log(self, message: str, *, min_verbose: int = 1) -> None:
        if int(self.config.verbose) >= int(min_verbose):
            self.logger.info(message)

    def calibration(self) -> None:
        """
        Run the full calibration workflow and compute distance-to-default.
        """
        self.historical_values_init()
        self.recurrent_estimation()

        if not self.asset_values:
            raise ValueError("Asset values not computed; calibration cannot complete.")

        self.distance_to_default = float(np.log(float(self.asset_values[-1]) / float(self.data.debt)))

        self._log(
            f"Calibration completed | iterations={self.iterations_number} | distance_to_default={self.distance_to_default}\n",
            min_verbose=1,
        )
        if int(self.config.verbose) >= 1:
            self._log(f"Default probability (%) = {self.default_probability_computation()}\n\n", min_verbose=1)

    def historical_values_init(self) -> None:
        """
        Initialize calibration by fitting physical parameters on equity log-returns and then updating risk-neutral parameters.
        """
        self._log("STEP 1 - historical_values_init\n\n", min_verbose=1)

        if self.model.parameters is None or self.model.risk_neutral_parameters is None:
            raise ValueError("model.parameters and model.risk_neutral_parameters must be initialized by the model.")

        param_names = list(getattr(self.model, "parameter_names", ()))
        if not param_names:
            raise ValueError("model.parameter_names must be defined for stable calibration.")

        init_phys = {name: float(self.model.parameters[name]) for name in param_names}
        self.physical_history.append(dict(init_phys))

        self.risk_neutral_history.append(dict(self.model.risk_neutral_parameters))

        equity_values = self.data.equity_array()
        equity_log_returns = np.diff(np.log(equity_values), n=1)

        initial_guess = np.asarray([init_phys[name] for name in param_names], dtype=float)
        self.physical_update(data=equity_log_returns, initial_guess=initial_guess)

        self.risk_neutral_update()


    def recurrent_estimation(self) -> None:
        """
        Iterate risk-neutral update, asset inference, and physical re-estimation until convergence
        or max_iterations is reached.
        """
        tolerance = float(self.config.tolerance)
        max_iterations = int(self.config.max_iterations)

        for iteration in range(1, max_iterations + 1):
            if self._physical_parameters_converged(tolerance=tolerance):
                break

            self.iterations_number = int(iteration)
            self._log(f"ITERATION {self.iterations_number}\n", min_verbose=1)

            self._log("STEP 2 - risk_neutral_update", min_verbose=1)
            self.risk_neutral_update()

            self._log("STEP 3 - asset_log_returns_computation", min_verbose=1)
            asset_log_returns = np.asarray(self.asset_log_returns_computation(), dtype=float).reshape(-1)

            if asset_log_returns.size < 2:
                raise RuntimeError("Asset log-returns are too short; calibration cannot proceed.")
            if not np.all(np.isfinite(asset_log_returns)):
                raise RuntimeError("Asset log-returns are not finite; calibration cannot proceed.")

            self._log("STEP 4 - physical_update", min_verbose=1)
            self._physical_update_with_rollback(asset_log_returns=asset_log_returns)

    def physical_update(self, data: np.ndarray, initial_guess: np.ndarray) -> None:
        """
        Update physical parameters using moment matching on the provided log-returns.

        parameters_convention_update() is kept for downstream analysis, but optimization always uses model.parameter_names.
        """
        x = np.asarray(data, dtype=float).reshape(-1)
        x0 = np.asarray(initial_guess, dtype=float).reshape(-1)

        param_names = list(getattr(self.model, "parameter_names", ()))
        if not param_names:
            raise ValueError("model.parameter_names must be defined for stable calibration.")

        self.moment_matcher.update_sample_cumulants(observations=x)
        updated_parameters = self.optimizer.estimate_parameters(initial_guess=x0)

        base_params = {name: float(updated_parameters[name]) for name in param_names}

        self.model.parameters = dict(base_params)
        self.physical_history.append(dict(base_params))

        self.model.check_bounds()

        self.sample_cumulants = dict(self.moment_matcher.sample_cumulants)

        # Keep convention update for output analysis (can add derived keys).
        self.model.parameters_convention_update()

        if int(self.config.verbose) >= 2:
            param_vector = np.asarray([base_params[name] for name in param_names], dtype=float)
            residuals_pct = self.moment_matcher.residuals(parameter_vector=param_vector, relative=True)
            self._log(f"STEP 4 - physical parameters: {base_params}", min_verbose=2)
            self._log(f"STEP 4 - sample cumulants   : {self.sample_cumulants}", min_verbose=2)
            self._log(f"STEP 4 - residuals (%)      : {residuals_pct}\n\n\n", min_verbose=2)


    def risk_neutral_update(self) -> None:
        """
        Solve for p_star and update model risk-neutral parameters with bounds validation.
        """
        try:
            p_star = float(self.esscher_solver.solve(p_initial=float(self.p_star)))
        except Exception as exc:
            raise ValueError(f"Esscher p_star solver failed: {exc}") from exc

        updated_rn_params = self.model.risk_neutral_parameters_update(p_star=float(p_star))
        self.model.risk_neutral_parameters = dict(updated_rn_params)
        self.risk_neutral_history.append(dict(updated_rn_params))
        self.p_star = float(p_star)

        self.model.check_bounds()
        self._check_model_pricer_consistency()
        self._log(f"STEP 2 - p_star (bounded solver) = {self.p_star}\n", min_verbose=1)

    def asset_log_returns_computation(self) -> np.ndarray:
        """
        Infer daily asset values and return their log-returns.
        """
        total_days = int(self.days_number)
        day_indices = list(range(total_days))

        self._log(
            f"STEP 3 - asset_log_returns_computation | RN params = {self.model.risk_neutral_parameters}\n",
            min_verbose=1,
        )

        inferred_assets = self.asset_inference.infer_asset_values(day_indices=day_indices)
        self.asset_values = [float(v) for v in inferred_assets]

        self.final_asset_residual_computation()
        asset_log_returns = np.diff(np.log(np.asarray(self.asset_values, dtype=float)), n=1)
        return asset_log_returns.astype(float)

    def final_asset_residual_computation(self) -> None:
        """
        Compute and store the pricing residual for the last inferred asset value at maturity.
        """
        if not self.asset_values:
            self.final_asset_residual = float("nan")
            return

        self.pricer.T = float(self.data.maturity)

        equity_series = self.data.equity_values
        try:
            last_equity_value = float(getattr(equity_series, "iloc")[-1])
        except Exception:
            last_equity_value = float(np.asarray(equity_series, dtype=float).reshape(-1)[-1])

        self.pricer.target_price = float(last_equity_value)
        self.final_asset_residual = float(self.pricer.price_residual(S0=float(self.asset_values[-1])))

        self._log(f"Final asset residual = {self.final_asset_residual}", min_verbose=1)

    def default_probability_computation(self) -> float:
        """
        Compute the default probability in percent using the physical distribution at the calibration horizon.
        """
        if not np.isfinite(self.distance_to_default):
            raise ValueError("distance_to_default is not set. Run calibration() first.")
        return float(self.model.cdf(cdf_input=-float(self.distance_to_default), t=float(self.data.maturity)) * 100.0)

    def final_residuals(self) -> Dict[str, object]:
        """
        Return final moment-matching residuals and final asset pricing residual.
        """
        if self.model.get_parameters() is None:
            raise ValueError("Model parameters are not available.")

        current_parameter_vector = np.asarray(list(self.model.get_parameters().values()), dtype=float)
        relative_residuals = self.moment_matcher.residuals(parameter_vector=current_parameter_vector, relative=True)
        cumulant_names = list(self.model.theoretical_cumulants().keys())

        cumulant_residuals = {
            f"{name}_residual_pct": float(residual) for name, residual in zip(cumulant_names, relative_residuals)
        }
        return {"cumulants": cumulant_residuals, "final_asset_residual": float(self.final_asset_residual)}

    def get_final_asset(self) -> float:
        if not self.asset_values:
            raise ValueError("Asset values not available. Run calibration() first.")
        return float(self.asset_values[-1])

    def get_number_iteration(self) -> int:
        return int(self.iterations_number)

    def _physical_parameters_converged(self, *, tolerance: float) -> bool:
        """
        Check convergence by comparing the last two physical parameter sets.
        """
        if len(self.physical_history) < 2:
            return False

        previous_params = self.physical_history[-2]
        current_params = self.physical_history[-1]

        for name in current_params.keys():
            if abs(float(current_params[name]) - float(previous_params[name])) > float(tolerance):
                return False
        return True

    def _risk_neutral_update_with_retries(self) -> None:
        """
        Run the risk-neutral update with multiple initial guesses and rollback on failures.
        """
        previous_p_star = float(self.p_star)
        previous_rn_params = dict(self.model.risk_neutral_parameters) if self.model.risk_neutral_parameters else None
        rn_history_size_before = int(len(self.risk_neutral_history))

        candidate_guesses = self._candidate_p_star_guesses(previous_p_star=previous_p_star)
        last_error: Optional[Exception] = None

        for guess in candidate_guesses:
            try:
                self.p_star = float(guess)
                self.risk_neutral_update()
                return
            except Exception as exc:
                last_error = exc
                self._rollback_risk_neutral_state(
                    previous_p_star=previous_p_star,
                    previous_rn_params=previous_rn_params,
                    rn_history_size_before=rn_history_size_before,
                )
                self._log(f"Risk-neutral update failed (p_star_start={guess}): {exc}", min_verbose=1)

        raise RuntimeError(
            "Risk-neutral update failed after multiple attempts; calibration stopped to avoid invalid parameters."
        ) from last_error

    def _candidate_p_star_guesses(self, *, previous_p_star: float) -> List[float]:
        """
        Build a list of initial guesses for p_star retries.
        """
        guesses = [
            float(previous_p_star),
            float(self.config.p_star_initial),
            0.0,
            0.1,
            -0.1,
            1.0,
            -1.0,
        ]

        seen = set()
        unique_guesses: List[float] = []
        for value in guesses:
            v = float(value)
            if v not in seen:
                seen.add(v)
                unique_guesses.append(v)
        return unique_guesses

    def _rollback_risk_neutral_state(
        self,
        *,
        previous_p_star: float,
        previous_rn_params: Optional[Dict[str, float]],
        rn_history_size_before: int,
    ) -> None:
        """
        Restore model risk-neutral state and truncate history to its previous size.
        """
        self.p_star = float(previous_p_star)
        if previous_rn_params is not None:
            self.model.risk_neutral_parameters = dict(previous_rn_params)
        while len(self.risk_neutral_history) > int(rn_history_size_before):
            self.risk_neutral_history.pop()

    def _asset_log_returns_checked(self) -> np.ndarray:
        """
        Compute asset log-returns and validate finiteness and minimum length.
        """
        asset_log_returns = np.asarray(self.asset_log_returns_computation(), dtype=float)
        if asset_log_returns.size < 2:
            raise RuntimeError("Asset log-returns are too short; calibration cannot proceed.")
        if not np.all(np.isfinite(asset_log_returns)):
            raise RuntimeError("Asset log-returns are not finite; calibration cannot proceed.")
        return asset_log_returns
    
    def _physical_update_with_rollback(self, *, asset_log_returns: np.ndarray) -> None:
        """
        Run physical_update and rollback model state and history if the update fails.
        """
        param_names = list(getattr(self.model, "parameter_names", ()))
        if not param_names:
            raise ValueError("model.parameter_names must be defined for stable calibration.")

        previous_params = dict(self.model.parameters) if self.model.parameters is not None else None
        history_size_before = int(len(self.physical_history))

        try:
            last_physical = dict(self.physical_history[-1])
            initial_guess = np.asarray([float(last_physical[name]) for name in param_names], dtype=float)

            self.physical_update(
                data=np.asarray(asset_log_returns, dtype=float).reshape(-1),
                initial_guess=initial_guess,
            )
        except Exception as exc:
            if previous_params is not None:
                self.model.parameters = dict(previous_params)
                self.model.parameters_convention_update()

            while len(self.physical_history) > history_size_before:
                self.physical_history.pop()

            raise RuntimeError(f"Physical parameter update failed: {exc}") from exc


    def _check_model_pricer_consistency(self) -> None:
        """
        Validate that the pricer and calibrator reference the same model instance.
        """
        if self.model is not self.pricer.model:
            raise ValueError("The model instance is not correctly shared with the pricer.")

    @staticmethod
    def _default_executor_kind(*, prefer_threads_on_windows: bool, configured_kind: str) -> str:
        """
        Resolve executor kind when configuration is set to 'auto'.
        """
        kind = str(configured_kind).lower()
        if kind != "auto":
            return kind
        if os.name == "nt" and bool(prefer_threads_on_windows):
            return "thread"
        return "process"
