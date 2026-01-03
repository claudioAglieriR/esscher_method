# esscher_method/model/model.py
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import scipy.integrate as integrate
from esscher_method.model.numerics import (
    DEFAULT_DOMAIN_CONFIG,
    DEFAULT_INTEGRATION_CONFIG,
    DomainConfig,
    IntegrationConfig,
)
from esscher_method.model.policies import (
    DEFAULT_DELTA,
    DEFAULT_BILATERAL_GAMMA_POLICY,
    DEFAULT_MERTON_POLICY,
    DEFAULT_VARIANCE_GAMMA_POLICY,
    BilateralGammaPolicy,
    MertonPolicy,
    VarianceGammaPolicy,
)


__all__ = [
    "Model",
    "Merton",
    "BilateralGamma",
    "VarianceGamma",
]


class Model(ABC):
    def __init__(
        self,
        delta: Optional[float] = None,
        parameters: Optional[Dict[str, float]] = None,
        integration: Optional[IntegrationConfig] = None,
        domain: Optional[DomainConfig] = None,
    ):
        self.parameters = parameters
        self.risk_neutral_parameters = parameters

        self.delta = DEFAULT_DELTA if delta is None else float(delta)

        self.integration = integration or DEFAULT_INTEGRATION_CONFIG
        self.domain = domain or DEFAULT_DOMAIN_CONFIG

        self.cgf_bounds = [()]
        self.bounds: list[tuple[float, float]] = []

        self.parameter_names: Tuple[str, ...] = tuple()
        self.risk_neutral_parameter_map: Dict[str, str] = {}
        self.cgf_lower_bound: float = -1.0e3
        self.cgf_upper_bound: float = 1.0e3

    @property
    def steps_per_year(self) -> float:
        """
        Sampling frequency implied by the model time step.
        Example: delta=1/252 -> steps_per_year=252.
        """
        return 1.0 / float(self.delta)
    
    def parameters_convention_update(self) -> None:
        return None



    @abstractmethod
    def theoretical_cumulants(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def parameters_update(self, parameters: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def cumulant_generating_function(self, cgf_input: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def risk_neutral_parameters_update(self, p_star: float) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def chf(self, chf_input: complex, t: float, risk_neutral: bool) -> complex:
        raise NotImplementedError

    def get_parameters(self) -> Optional[Dict[str, float]]:
        return self.parameters


    def theoretical_cumulants_update(self, parameters: np.ndarray) -> Dict[str, float]:
        self.parameters_update(parameters=parameters)
        return self.theoretical_cumulants()
    
    def cdf(
        self,
        cdf_input: float,
        t: float,
        limit: Optional[int] = None,
        upper_bound: Optional[float] = None,
        singularity_tolerance: Optional[float] = None,
        config: Optional[IntegrationConfig] = None,
    ) -> float:
        """
        Compute the cumulative distribution function at 'cdf_input' for horizon 't'
        using the Gil-Pelaez inversion formula and the model physical characteristic function.

        Numerical settings (quadrature limit, integration upper bound, and the small offset used
        to avoid the singularity at u = 0) are taken from 'config' when provided.

        Raises:
            RuntimeError: if the numerical integration fails.
            ValueError: if the computed integral is not finite.
        """
        cfg = config or self.integration
        quad_limit = cfg.cdf_quad_limit if limit is None else int(limit)
        ub = cfg.cdf_upper_bound if upper_bound is None else float(upper_bound)
        tol = cfg.cdf_singularity_tol if singularity_tolerance is None else float(singularity_tolerance)

        cdf_input = float(cdf_input)
        t = float(t)

        def integrand(u: float) -> float:
            val = np.exp(-1j * u * cdf_input) * self.chf(chf_input=u, t=t, risk_neutral=False) / (1j * u)
            return float(np.real(val))

        try:
            integral = integrate.quad(integrand, a=tol, b=ub, limit=quad_limit)[0]
        except Exception as exc:
            raise RuntimeError("Failed to compute CDF via Gil-Pelaez inversion.") from exc

        if not np.isfinite(integral):
            raise ValueError("CDF integral returned a non-finite value.")

        return 0.5 - (1.0 / math.pi) * float(integral)

    def _require_parameters(self) -> None:
        """
        Validate that both physical and risk-neutral parameter dictionaries are available.

        Raises:
            ValueError: if 'self.parameters' or 'self.risk_neutral_parameters' is None.
        """
        if self.parameters is None:
            raise ValueError("parameters must not be None.")
        if self.risk_neutral_parameters is None:
            raise ValueError("risk_neutral_parameters must not be None.")

    def _bounds_by_name(self) -> Dict[str, Tuple[float, float]]:
        """
        Build a mapping from parameter name to (lower_bound, upper_bound).

        If 'self.parameter_names' is empty, it is inferred from 'self.parameters' keys.
        The method validates that the number of bounds matches the number of parameter names.

        Returns:
            A dict mapping each parameter name to its bounds.

        Raises:
            ValueError: if 'parameter_names' is empty and 'parameters' is None,
                        or if the bounds length does not match 'parameter_names' length.
        """
        if not self.parameter_names:
            if self.parameters is None:
                raise ValueError("parameter_names is empty and parameters is None.")
            self.parameter_names = tuple(self.parameters.keys())

        if len(self.bounds) != len(self.parameter_names):
            raise ValueError(
                f"bounds length ({len(self.bounds)}) does not match "
                f"parameter_names length ({len(self.parameter_names)})."
            )

        return dict(zip(self.parameter_names, self.bounds))

    @staticmethod
    def _check_named_params(
        params: Mapping[str, float],
        bounds_by_name: Mapping[str, Tuple[float, float]],
        names: Iterable[str],
        label: str,
    ) -> None:
        """
        Validate that a set of named parameters exists and lies within configured bounds.

        Raises:
            ValueError: if a parameter is missing, missing bounds, or out of bounds.
        """
        for name in names:
            if name not in params:
                raise ValueError(f"Missing {label} parameter '{name}'.")
            if name not in bounds_by_name:
                raise ValueError(f"Missing bounds for parameter '{name}'.")

            value = float(params[name])
            lower, upper = bounds_by_name[name]
            if value < lower or value > upper:
                raise ValueError(
                    f"Parameter '{name}' out of bounds for {label}: {value} not in [{lower}, {upper}]."
                )

    def _risk_neutral_names_and_bounds(
        self, bounds_by_name: Mapping[str, Tuple[float, float]]
    ) -> Tuple[Tuple[str, ...], Dict[str, Tuple[float, float]]]:
        """
        Determine which risk-neutral parameter names should be validated and the bounds to use.

        Returns:
            A tuple with:
              - rn_names: risk-neutral parameter names to validate
              - rn_bounds: bounds mapping for those risk-neutral names

        Raises:
            ValueError: if a mapped physical parameter does not have bounds in 'bounds_by_name'.
        """
        if self.risk_neutral_parameter_map:
            rn_names = tuple(self.risk_neutral_parameter_map.keys())
            rn_bounds: Dict[str, Tuple[float, float]] = {}
            for rn_name, phys_name in self.risk_neutral_parameter_map.items():
                if phys_name not in bounds_by_name:
                    raise ValueError(f"Missing bounds for physical parameter '{phys_name}'.")
                rn_bounds[rn_name] = bounds_by_name[phys_name]
            return rn_names, rn_bounds

        rn_names = tuple(name for name in self.parameter_names if name in (self.risk_neutral_parameters or {}))
        rn_bounds = {name: bounds_by_name[name] for name in rn_names}
        return rn_names, rn_bounds

    def check_bounds(self) -> None:
        """
        Validate physical and risk-neutral parameters against the model bounds.

        The method checks:
          - that parameter dictionaries exist
          - that physical parameters listed in 'self.parameter_names' exist and are within bounds
          - that risk-neutral parameters (as defined by 'risk_neutral_parameter_map' or inferred)
            exist and are within the appropriate bounds

        Raises:
            ValueError: if parameters are missing, bounds are missing, or any parameter is out of bounds.
        """
        self._require_parameters()
        bounds_by_name = self._bounds_by_name()

        self._check_named_params(
            params=self.parameters,
            bounds_by_name=bounds_by_name,
            names=self.parameter_names,
            label="physical",
        )

        rn_names, rn_bounds = self._risk_neutral_names_and_bounds(bounds_by_name=bounds_by_name)

        self._check_named_params(
            params=self.risk_neutral_parameters,
            bounds_by_name=rn_bounds,
            names=rn_names,
            label="risk-neutral",
        )
    def cgf_domain_bounds(self, eps: float = 1.0e-12) -> tuple[float, float]:
        """
        Return the mathematical domain (lower, upper) where the cumulant generating function is defined
        for real inputs.

        Default implementation: assume the CGF is defined on the whole real line.
        Subclasses should override when the CGF has singularities / finite domain.
        """
        return (-float("inf"), float("inf"))

    def update_cgf_bounds(
        self,
        eps: float = 1.0e-8,
        default_span: float = 1.0e3,
    ) -> tuple[float, float]:
        """
        Compute and store a finite interval [cgf_lower_bound, cgf_upper_bound] suitable for numerically
        solving the Esscher equation.

        This method guarantees:
          - cgf_lower_bound and cgf_upper_bound always exist
          - the returned bounds are finite floats
        """
        lower_dom, upper_dom = self.cgf_domain_bounds(eps=eps)

        lower_p = float(lower_dom)
        upper_p = float(upper_dom - 1.0) if np.isfinite(upper_dom) else float("inf")

        if not np.isfinite(lower_p):
            lower_p = -float(default_span)
        if not np.isfinite(upper_p):
            upper_p = float(default_span)

        # Avoid degenerate / invalid intervals.
        if not (lower_p < upper_p):
            raise ValueError(
                f"Invalid CGF bounds for Esscher solver: lower={lower_p}, upper={upper_p}."
            )

        self.cgf_lower_bound = float(lower_p)
        self.cgf_upper_bound = float(upper_p)
        return (self.cgf_lower_bound, self.cgf_upper_bound)
    
    def esscher_p_star_bounds(self, eps: float = 1.0e-8, default_span: float = 1.0e3) -> tuple[float, float]:
        """
        Return a finite interval [lower, upper] where the Esscher equation should be solved.

        Default behaviour: use update_cgf_bounds(), which already enforces that both p and (p+1)
        are in the CGF domain (via the '-1' on the upper bound).
        Subclasses further restrict this interval to ensure risk-neutral parameters stay within bounds.
        """
        return self.update_cgf_bounds(eps=float(eps), default_span=float(default_span))




class Merton(Model):
    def __init__(
        self,
        delta: Optional[float] = None,
        parameters: Optional[Dict[str, float]] = None,
        policy: Optional[MertonPolicy] = None,
        integration: Optional[IntegrationConfig] = None,
        domain: Optional[DomainConfig] = None,
    ):
        super().__init__(
            delta=delta,
            parameters=parameters,
            integration=integration,
            domain=domain,
        )

        self.policy = policy or DEFAULT_MERTON_POLICY
        self.parameter_names = ("mu", "sigma")
        # because of its simplicity, the risk-neutral case in the Merton model is handled
        # through the two methods _gaussian_chf and chf
        self.risk_neutral_parameter_map = {}

        if parameters is None:
            parameters = dict(self.policy.default_parameters)

        self.parameters = {k: float(parameters[k]) for k in self.parameter_names}
        self.bounds = self.policy.bounds()

        self.bounds_intermediate = self.policy.intermediate_bounds(
            mu=float(self.parameters["mu"]),
            sigma=float(self.parameters["sigma"]),
        )
        self.bounds_initialization = self.bounds_intermediate

        self.risk_neutral_parameters = self.risk_neutral_parameters_update(p_star=0.0)

    @property
    def mu(self) -> float:
        return float(self.parameters["mu"])

    @mu.setter
    def mu(self, value: float) -> None:
        self.parameters["mu"] = float(value)

    @property
    def sigma(self) -> float:
        return float(self.parameters["sigma"])

    @sigma.setter
    def sigma(self, value: float) -> None:
        self.parameters["sigma"] = float(value)

    def theoretical_cumulants(self) -> Dict[str, float]:
        return {
            "mean": self.mu * self.delta,
            "variance": (self.sigma**2) * self.delta,
        }

    def parameters_update(self, parameters: np.ndarray) -> None:
        self.mu = float(parameters[0])
        self.sigma = float(parameters[1])

    def cumulant_generating_function(self, cgf_input: float) -> float:
        u = float(cgf_input)
        return float((self.mu * u + 0.5 * (self.sigma**2) * (u**2)) * self.delta)


    def risk_neutral_parameters_update(self, p_star: float) -> Dict[str, float]:
        p = float(p_star)
        mu_rn = self.mu + (self.sigma**2) * p
        return {"mu": float(mu_rn), "sigma": float(self.sigma)}

    @staticmethod
    def _gaussian_chf(mu: float, sigma: float, u: complex, t: float) -> complex:
        return np.exp((1j * mu * t * u) - 0.5 * (sigma**2) * (u**2) * t)

    def chf(self, chf_input: complex, t: float, risk_neutral: bool) -> complex:
        tt = float(t)
        u = complex(chf_input)
        if risk_neutral:
            params = self.risk_neutral_parameters or {"mu": self.mu, "sigma": self.sigma}
            return self._gaussian_chf(float(params["mu"]), float(params["sigma"]), u, tt)
        return self._gaussian_chf(self.mu, self.sigma, u, tt)



class BilateralGamma(Model):
    def __init__(
        self,
        delta: Optional[float] = None,
        parameters: Optional[Dict[str, float]] = None,
        policy: Optional[BilateralGammaPolicy] = None,
        integration: Optional[IntegrationConfig] = None,
        domain: Optional[DomainConfig] = None,
    ):
        super().__init__(
            delta=delta,
            parameters=parameters,
            integration=integration,
            domain=domain,
        )

        self.policy = policy or DEFAULT_BILATERAL_GAMMA_POLICY
        if parameters is None:
            parameters = dict(self.policy.default_parameters)

        self.parameters = parameters
        self.risk_neutral_parameters = self.risk_neutral_parameters_update(p_star=0.0)

        self.parameter_names = ("alpha_P", "lambda_P", "alpha_M", "lambda_M")
        self.risk_neutral_parameter_map = {
            "alpha_P_RN": "alpha_P",
            "lambda_P_RN": "lambda_P",
            "alpha_M_RN": "alpha_M",
            "lambda_M_RN": "lambda_M",
        }
        self.bounds = self.policy.bounds()
        self.update_cgf_bounds()

    @property
    def alpha_P(self) -> float:
        return float(self.parameters["alpha_P"])

    @alpha_P.setter
    def alpha_P(self, value: float) -> None:
        self.parameters["alpha_P"] = float(value)

    @property
    def lambda_P(self) -> float:
        return float(self.parameters["lambda_P"])

    @lambda_P.setter
    def lambda_P(self, value: float) -> None:
        self.parameters["lambda_P"] = float(value)

    @property
    def alpha_M(self) -> float:
        return float(self.parameters["alpha_M"])

    @alpha_M.setter
    def alpha_M(self, value: float) -> None:
        self.parameters["alpha_M"] = float(value)

    @property
    def lambda_M(self) -> float:
        return float(self.parameters["lambda_M"])

    @lambda_M.setter
    def lambda_M(self, value: float) -> None:
        self.parameters["lambda_M"] = float(value)

    def _bg_params(self, risk_neutral: bool) -> Tuple[float, float, float, float]:
        self._require_parameters()
        if risk_neutral:
            p = self.risk_neutral_parameters
            return (
                float(p["alpha_P_RN"]),
                float(p["lambda_P_RN"]),
                float(p["alpha_M_RN"]),
                float(p["lambda_M_RN"]),
            )
        return self.alpha_P, self.lambda_P, self.alpha_M, self.lambda_M
    def update_cgf_bounds(self, eps: float = 1.0e-8, default_span: float = 1.0e3) -> tuple[float, float]:
        """
        Update both:
        - self.cgf_bounds: domain for CGF evaluation at real inputs (p)
        - self.cgf_lower_bound/self.cgf_upper_bound: safe bounds for Esscher p, ensuring p+1 is also admissible
        """
        eps = float(self.domain.cgf_domain_eps) if eps is None else float(eps)

        lower_dom, upper_dom = self.cgf_domain_bounds(eps=eps)

        self.cgf_bounds = [(float(lower_dom), float(upper_dom))]

        lower_p = float(lower_dom)
        upper_p = float(upper_dom - 1.0)

        if not (np.isfinite(lower_p) and np.isfinite(upper_p) and lower_p < upper_p):
            raise ValueError(f"Invalid CGF bounds for Esscher: lower={lower_p}, upper={upper_p}.")

        self.cgf_lower_bound = lower_p
        self.cgf_upper_bound = upper_p
        return (self.cgf_lower_bound, self.cgf_upper_bound)
    
    def esscher_p_star_bounds(self, eps: float = 1.0e-8, default_span: float = 1.0e3) -> tuple[float, float]:
        """
        Restrict Esscher bounds further so that RN lambdas remain within configured bounds
        """
        lower, upper = self.update_cgf_bounds(eps=float(eps), default_span=float(default_span))

        bounds_by_name = self._bounds_by_name()
        lo_lp, hi_lp = bounds_by_name["lambda_P"]
        lo_lm, hi_lm = bounds_by_name["lambda_M"]

        lambda_p = float(self.lambda_P)
        lambda_m = float(self.lambda_M)

        # From lambda_P - p in [lo_lp, hi_lp]
        lower = max(lower, lambda_p - float(hi_lp))
        upper = min(upper, lambda_p - float(lo_lp))

        # From lambda_M + p in [lo_lm, hi_lm]
        lower = max(lower, float(lo_lm) - lambda_m)
        upper = min(upper, float(hi_lm) - lambda_m)

        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise ValueError(f"Empty Esscher feasible interval after RN-bounds restriction: lower={lower}, upper={upper}.")

        return (float(lower), float(upper))


    def theoretical_cumulants(self) -> Dict[str, float]:
        return {
            "mean": self.cumulant(n=1),
            "variance": self.cumulant(n=2),
            "skewness": self.cumulant(n=3),
            "fourth_cumulant": self.cumulant(n=4),
        }

    def parameters_update(self, parameters: np.ndarray) -> None:
        self.alpha_P = float(parameters[0])
        self.lambda_P = float(parameters[1])
        self.alpha_M = float(parameters[2])
        self.lambda_M = float(parameters[3])

    def cumulant(self, n: int) -> float:
        alpha_p, lambda_p, alpha_m, lambda_m = self._bg_params(risk_neutral=False)
        return (
            math.factorial(n - 1)
            * (alpha_p / (lambda_p**n) + ((-1) ** n) * alpha_m / (lambda_m**n))
            * self.delta
        )

    def cumulant_generating_function(self, cgf_input: float) -> float:
        u = float(np.asarray(cgf_input, dtype=float))
        alpha_p, lambda_p, alpha_m, lambda_m = self._bg_params(risk_neutral=False)

        eps = float(self.domain.cgf_domain_eps) if hasattr(self, "domain") else 1e-12
        if (u <= -lambda_m + eps) or (u >= lambda_p - eps):
            return float("nan")

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            k = (-alpha_p * np.log1p(-u / lambda_p)) + (-alpha_m * np.log1p(u / lambda_m))

        k = float(k) * float(self.delta)
        return k if np.isfinite(k) else float("nan")


    def p_star_minimizer(self, cgf_input: float) -> float:
        p = float(np.asarray(cgf_input, dtype=float))
        alpha_p, lambda_p, alpha_m, lambda_m = self._bg_params(risk_neutral=False)

        eps = float(self.domain.cgf_domain_eps) if hasattr(self, "domain") else 1e-12

        if (p <= -lambda_m + eps) or (p >= lambda_p - eps):
            return float("nan")
        if ((p + 1.0) <= -lambda_m + eps) or ((p + 1.0) >= lambda_p - eps):
            return float("nan")

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            Kp   = (-alpha_p * np.log1p(-p / lambda_p)) + (-alpha_m * np.log1p(p / lambda_m))
            Kp1  = (-alpha_p * np.log1p(-(p + 1.0) / lambda_p)) + (-alpha_m * np.log1p((p + 1.0) / lambda_m))

            ratio = np.exp(Kp1 - Kp)   # = M(p+1)/M(p)

        ratio = float(ratio)
        if not np.isfinite(ratio):
            return float("nan")

        return ratio - 1.0


    def absolute_p_star_minimizer(self, cgf_input: float) -> float:
        return float(np.abs(self.p_star_minimizer(cgf_input=cgf_input)))

    def exponential_convexity_correction(self) -> float:
        alpha_p, lambda_p, alpha_m, lambda_m = self._bg_params(risk_neutral=False)

        if lambda_p <= 1.0:
            return float("nan")

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            log_c = (-alpha_p * np.log1p(-1.0 / lambda_p)) + (-alpha_m * np.log1p(1.0 / lambda_m))
            c = np.exp(log_c)

        c = float(c)
        return c if np.isfinite(c) else float("nan")


    def risk_neutral_parameters_update(self, p_star: float) -> Dict[str, float]:
        p_star = float(p_star)
        return {
            "alpha_P_RN": self.alpha_P,
            "lambda_P_RN": self.lambda_P - p_star,
            "alpha_M_RN": self.alpha_M,
            "lambda_M_RN": self.lambda_M + p_star,
        }

    def chf(self, chf_input: complex, t: float, risk_neutral: bool) -> complex:
        alpha_p, lambda_p, alpha_m, lambda_m = self._bg_params(risk_neutral=risk_neutral)
        return (lambda_p / (lambda_p - 1j * chf_input)) ** (alpha_p * t) * (lambda_m / (lambda_m + 1j * chf_input)) ** (
            alpha_m * t
        )
        
    def cgf_domain_bounds(self, eps: float = 1.0e-12) -> tuple[float, float]:
        """
        For a bilateral gamma CGF, singularities typically occur at:
          - p =  lambda_P
          - p = -lambda_M
        so the real domain is approximately (-lambda_M, lambda_P).
        """
        params = self.parameters or {}
        lambda_p = float(params.get("lambda_P"))
        lambda_m = float(params.get("lambda_M"))

        if lambda_p <= 0.0 or lambda_m <= 0.0:
            raise ValueError("Invalid BilateralGamma parameters: lambda_P and lambda_M must be positive.")

        lower = -lambda_m + float(eps)
        upper = lambda_p - float(eps)
        return (lower, upper)


class VarianceGamma(BilateralGamma):
    def __init__(
        self,
        delta: Optional[float] = None,
        parameters: Optional[Dict[str, float]] = None,
        policy: Optional[VarianceGammaPolicy] = None,
        integration: Optional[IntegrationConfig] = None,
        domain: Optional[DomainConfig] = None,
    ):
        Model.__init__(
            self,
            delta=delta,
            parameters=parameters,
            integration=integration,
            domain=domain,
        )

        self.policy = policy or DEFAULT_VARIANCE_GAMMA_POLICY
        if parameters is None:
            parameters = dict(self.policy.default_parameters)

        self.parameters = parameters
        self.risk_neutral_parameters = self.risk_neutral_parameters_update(p_star=0.0)

        self.parameter_names = ("alpha", "lambda_P", "lambda_M")
        self.risk_neutral_parameter_map = {
            "alpha_RN": "alpha",
            "lambda_P_RN": "lambda_P",
            "lambda_M_RN": "lambda_M",
        }
        self.bounds = self.policy.bounds()
        self.update_cgf_bounds()

    @property
    def alpha(self) -> float:
        return float(self.parameters["alpha"])

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.parameters["alpha"] = float(value)

    @property
    def lambda_P(self) -> float:
        return float(self.parameters["lambda_P"])

    @lambda_P.setter
    def lambda_P(self, value: float) -> None:
        self.parameters["lambda_P"] = float(value)

    @property
    def lambda_M(self) -> float:
        return float(self.parameters["lambda_M"])

    @lambda_M.setter
    def lambda_M(self, value: float) -> None:
        self.parameters["lambda_M"] = float(value)

    def _bg_params(self, risk_neutral: bool) -> Tuple[float, float, float, float]:
        self._require_parameters()
        if risk_neutral:
            p = self.risk_neutral_parameters
            a = float(p["alpha_RN"])
            return a, float(p["lambda_P_RN"]), a, float(p["lambda_M_RN"])
        a = self.alpha
        return a, self.lambda_P, a, self.lambda_M

    def theoretical_cumulants(self) -> Dict[str, float]:
        return {
            "mean": self.cumulant(n=1),
            "variance": self.cumulant(n=2),
            "fourth_cumulant": self.cumulant(n=4),
        }

    def parameters_update(self, parameters: np.ndarray) -> None:
        self.alpha = float(parameters[0])
        self.lambda_P = float(parameters[1])
        self.lambda_M = float(parameters[2])

    def risk_neutral_parameters_update(self, p_star: float) -> Dict[str, float]:
        p_star = float(p_star)
        return {
            "alpha_RN": self.alpha,
            "lambda_P_RN": self.lambda_P - p_star,
            "lambda_M_RN": self.lambda_M + p_star,
        }

    def parameters_convention_update(self) -> None:
        alpha = self.alpha
        lambda_p = self.lambda_P
        lambda_m = self.lambda_M
        self.parameters["sigma"] = float(np.sqrt((2.0 * alpha) / (lambda_p * lambda_m)))
        self.parameters["theta"] = float(alpha / lambda_p - alpha / lambda_m)
        self.parameters["nu"] = float(1.0 / alpha)
        
      
        
        
    
    def cgf_domain_bounds(self, eps: float = 1.0e-12) -> tuple[float, float]:
        """
        VarianceGamma is initially parameterized with (alpha, lambda_P, lambda_M),
        so the CGF real domain is (-lambda_M, lambda_P).
        """
        params = self.parameters or {}

        if "lambda_P" in params and "lambda_M" in params:
            lambda_p = float(params["lambda_P"])
            lambda_m = float(params["lambda_M"])
            if lambda_p <= 0.0 or lambda_m <= 0.0:
                raise ValueError("Invalid VarianceGamma parameters: lambda_P and lambda_M must be positive.")
            return (-lambda_m + float(eps), lambda_p - float(eps))

        # Optional fallback for alternative parametrizations if ever used:
        if all(k in params for k in ("sigma", "theta", "nu")):
            sigma = float(params["sigma"])
            theta = float(params["theta"])
            nu = float(params["nu"])
            if sigma <= 0.0 or nu <= 0.0:
                raise ValueError("Invalid VarianceGamma parameters: sigma and nu must be positive.")

            a = 0.5 * sigma * sigma * nu
            b = theta * nu
            c = -1.0
            disc = b * b - 4.0 * a * c
            if disc <= 0.0:
                raise ValueError("Invalid VarianceGamma parameters: discriminant must be positive.")
            sqrt_disc = float(np.sqrt(disc))
            r1 = (-b - sqrt_disc) / (2.0 * a)
            r2 = (-b + sqrt_disc) / (2.0 * a)
            lower = min(r1, r2) + float(eps)
            upper = max(r1, r2) - float(eps)
            return (lower, upper)

        return (-float("inf"), float("inf"))
