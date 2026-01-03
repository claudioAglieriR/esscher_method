# esscher_method/calibrator/moment_matching.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Mapping, Iterator

import numpy as np
from scipy.stats import moment

from esscher_method.model.model import Model
from .bounds import bounds_penalty
from .data_calibration import CalibrationConfig



@dataclass(frozen=True)
class SampleCumulants(Mapping[str, float]):
    """
    Read-only mapping wrapper for sample cumulants.

    Implementing Mapping allows dict(sample_cumulants) and iteration over keys.
    """
    values: Dict[str, float]

    def __getitem__(self, key: str) -> float:
        return float(self.values[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return int(len(self.values))

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.values.get(name, default))
    
class MomentMatcher:
    """
    Compute sample cumulants from observations and provide objective/residual functions
    for moment matching against model theoretical cumulants.
    """

    def __init__(self, model: Model, config: CalibrationConfig) -> None:
        if model is None:
            raise ValueError("model must not be None.")
        if config is None:
            raise ValueError("config must not be None.")

        self.model = model
        self.config = config
        self.sample_cumulants = SampleCumulants(values={})

    def required_cumulant_names(self) -> Sequence[str]:
        """
        Return cumulant names required by the model's theoretical cumulants.
        """
        theo = self.model.theoretical_cumulants()
        return tuple(theo.keys())

    def update_sample_cumulants(self, observations: np.ndarray) -> None:
        """
        Compute unbiased sample cumulants from 1D observations.

        The required minimum sample size depends on the cumulants required by the model:
          - mean/variance: at least 2 observations
          - skewness/fourth_cumulant: at least 5 observations for stability and unbiased correction
        """
        observations_array = np.asarray(observations, dtype=float).reshape(-1)
        n = int(observations_array.size)
        if n < 2:
            raise ValueError("Not enough observations to estimate mean and variance.")

        required = set(self.required_cumulant_names())
        needs_higher = ("skewness" in required) or ("fourth_cumulant" in required)
        if needs_higher and n < 5:
            raise ValueError("Not enough observations to estimate higher-order cumulants.")

        out: Dict[str, float] = {}
        out["mean"] = float(np.mean(observations_array))

        # Unbiased sample variance
        out["variance"] = float(np.var(observations_array, ddof=1))

        if needs_higher:
            m2 = float(moment(observations_array, moment=2))
            m3 = float(moment(observations_array, moment=3))
            m4 = float(moment(observations_array, moment=4))

            # Unbiased third central moment 
            out["skewness"] = float((n * n) / ((n - 1) * (n - 2)) * m3)

            # Unbiased fourth cumulant 
            out["fourth_cumulant"] = float(
                (n * n) / ((n - 1) * (n - 2) * (n - 3))
                * ((n + 1) * m4 - 3.0 * (n - 1) * (m2 * m2))
            )

        self.sample_cumulants = SampleCumulants(values=out)

    def objective(self, parameter_vector: np.ndarray) -> float:
        """
        Weighted relative deviation between theoretical and sample cumulants with a soft bounds penalty.
        """
        parameters_reshaped = np.asarray(parameter_vector, dtype=float).reshape(-1)

        theoretical = self.model.theoretical_cumulants_update(parameters=parameters_reshaped)
        theo_vals = np.asarray(list(theoretical.values()), dtype=float)

        if theo_vals.size == 0 or (not np.all(np.isfinite(theo_vals))):
            return 1e12

        weights = self.config.cumulant_weights or {}
        eps = 1e-12
        total = 0.0

        for name, theo_val in theoretical.items():
            w = float(weights.get(name, 1.0))
            sample_val = float(self.sample_cumulants.get(name, 0.0))
            denom = max(abs(sample_val), eps)
            total += w * abs((float(theo_val) - sample_val) / denom)

        bnds = getattr(self.model, "bounds", None) or []
        total += 1e-2 * bounds_penalty(parameters_reshaped, bnds)

        if not np.isfinite(total):
            return 1e12
        return float(total)

    def residuals(self, *, parameter_vector: np.ndarray, relative: bool = False) -> np.ndarray:
        """
        Return absolute or relative residuals in the order of model theoretical cumulants.
        """
        parameters_reshaped = np.asarray(parameter_vector, dtype=float).reshape(-1)
        theoretical = self.model.theoretical_cumulants_update(parameters=parameters_reshaped)

        names = list(theoretical.keys())
        diffs = np.array(
            [float(theoretical[name]) - float(self.sample_cumulants.get(name, 0.0)) for name in names],
            dtype=float,
        )

        if not relative:
            return diffs

        eps = 1e-12
        rel = []
        for name, d in zip(names, diffs):
            denom = max(abs(float(self.sample_cumulants.get(name, 0.0))), eps)
            rel.append(abs(float(d) / denom) * 100.0)
        return np.asarray(rel, dtype=float)

    def residual_vector(self, parameter_vector: np.ndarray) -> np.ndarray:
        """
        Residual vector for bounded least-squares solvers.

        Residuals are scaled as weighted relative errors to keep magnitudes comparable.
        """
        parameters_reshaped = np.asarray(parameter_vector, dtype=float).reshape(-1)
        theoretical = self.model.theoretical_cumulants_update(parameters=parameters_reshaped)

        weights = self.config.cumulant_weights or {}
        eps = 1e-12

        residuals: list[float] = []
        for name, theo_val in theoretical.items():
            sample_val = float(self.sample_cumulants.get(name, 0.0))
            denom = max(abs(sample_val), eps)
            w = float(weights.get(name, 1.0)) ** 0.5
            r = w * (float(theo_val) - sample_val) / denom
            if not np.isfinite(r):
                r = 1e6
            residuals.append(float(r))

        return np.asarray(residuals, dtype=float)
