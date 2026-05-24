"""
Equivalent martingale measures (EMM) as injectable strategies.

The structural Levy model lives in an incomplete market, so the choice of
EMM that maps the physical measure to a risk-neutral one is intrinsically
arbitrary. This module makes the choice explicit and replaceable via a small ABC + concrete strategies.

The default is EsscherMeasure, which reproduces the methodology of the
paper (Aguilar, Kirkby, Aglieri Rinella, JFI 2024) bit-by-bit. An
alternative MeanCorrectingMeasure is shipped as a Merton-only prototype
to expose the abstraction; for Merton it coincides algebraically with
Esscher (the Gaussian degenerate case, paper Remark 3 / Eq. 15), and on
BG / VG it raises NotImplementedError with a clear message.

Historical reference for the Esscher transform: Esscher, F. (1932). On the
probability function in the collective theory of risk. Skandinavisk
Aktuarietidskrift, 15, 175-195. Applied to option pricing in Gerber and
Shiu (1994), Trans. Soc. Actuaries, 46, 99-191.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from esscher_method.model.model import Merton, Model

from .esscher_solver import EsscherSolver, EsscherSolverConfig


class MartingaleMeasure(ABC):
    """
    Strategy interface for computing risk-neutral parameters of a Levy model
    given the physical parameters and the risk-free rate.

    Concrete implementations encapsulate one specific EMM. The two-step
    contract (solve -> apply) keeps the EMM-specific scalar parameter
    explicit and lets callers store / log it (e.g. Esscher's p_star) without
    inspecting model internals.
    """

    @abstractmethod
    def solve(
        self,
        *,
        model: Model,
        risk_free_rate: float,
        delta: float,
        p_initial: float = 0.0,
    ) -> float:
        """
        Compute the scalar EMM parameter. Semantic is EMM-specific:
        - EsscherMeasure: returns p_star, root of K_X(p+1) - K_X(p) = r.
        - MeanCorrectingMeasure: returns the additive drift shift such
          that K_X(1) under the shifted measure equals r.

        p_initial is an optional warm-start used only when the EMM relies
        on an iterative root finder (Esscher); closed-form EMMs ignore it.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, *, model: Model, emm_param: float) -> Dict[str, float]:
        """
        Translate the EMM parameter into the model's risk-neutral parameter
        dictionary, ready to be assigned to model.risk_neutral_parameters.
        """
        raise NotImplementedError


class EsscherMeasure(MartingaleMeasure):
    """
    Esscher transform EMM: p_star solves K_X(p+1) - K_X(p) = r (per unit
    time). Reproduces the methodology of the paper and is the library
    default; any new CalibrationConfig() instantiates an EsscherMeasure
    with the default EsscherSolverConfig.

    The optional config parameter forwards to EsscherSolver, exposing the
    grid / brentq / least-squares tolerances for callers who need to tune
    them. Default config=None preserves the paper-baseline numerics.
    """

    def __init__(self, config: Optional[EsscherSolverConfig] = None) -> None:
        self._config = config

    @property
    def config(self) -> Optional[EsscherSolverConfig]:
        return self._config

    def solve(
        self,
        *,
        model: Model,
        risk_free_rate: float,
        delta: float,
        p_initial: float = 0.0,
    ) -> float:
        solver = EsscherSolver(
            model=model,
            risk_free_rate=float(risk_free_rate),
            delta=float(delta),
            config=self._config,
        )
        return float(solver.solve(p_initial=float(p_initial)))

    def apply(self, *, model: Model, emm_param: float) -> Dict[str, float]:
        return dict(model.risk_neutral_parameters_update(p_star=float(emm_param)))


class MeanCorrectingMeasure(MartingaleMeasure):
    """
    Mean-correcting EMM: shifts the drift of X so that the discounted asset
    e^{-rt} V_A(t) is a martingale, i.e., K_X(1) under the shifted measure
    equals r (per unit time). Closed-form per model.

    Prototype scope: Merton only. For Merton the shifted measure coincides
    with Esscher (paper Remark 3 / Eq. 15: mu_RN = r - sigma^2/2 regardless
    of method); the abstraction exists to make EMM arbitrariness an
    explicit decision in the API. For BG / VG the drift shift is
    mathematically well-defined whenever K_X(1) is finite (i.e., always
    inside the policy-enforced admissible bounds) but its translation to
    the native (alpha, lambda_P, lambda_M) parameters is non-trivial and
    pending future work; apply() raises NotImplementedError with a clear
    message in that case.
    """

    def solve(
        self,
        *,
        model: Model,
        risk_free_rate: float,
        delta: float,
        p_initial: float = 0.0,
    ) -> float:
        # model.cumulant_generating_function returns the delta-scaled CGF
        # delta * K_X(u); divide by delta to get the per-unit-time value
        # K_X(1) used by the mean-correcting condition K_X(1) = r.
        cgf_at_1 = float(model.cumulant_generating_function(cgf_input=1.0))
        kx_at_1_per_unit_time = cgf_at_1 / float(delta)
        return float(risk_free_rate) - kx_at_1_per_unit_time

    def apply(self, *, model: Model, emm_param: float) -> Dict[str, float]:
        if not isinstance(model, Merton):
            raise NotImplementedError(
                f"MeanCorrectingMeasure currently implements only Merton "
                f"(received {type(model).__name__}). The drift shift is "
                f"mathematically well-defined for any Levy model with finite "
                f"exponential moment of order 1, but its translation to the "
                f"native (alpha, lambda_P, lambda_M) parameters of BG / VG "
                f"is non-trivial and pending future work. Use EsscherMeasure "
                f"(the library default) for BG / VG."
            )
        drift_shift = float(emm_param)
        return {"mu": float(model.mu) + drift_shift, "sigma": float(model.sigma)}
