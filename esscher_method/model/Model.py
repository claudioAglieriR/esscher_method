import math
from itertools import chain
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.integrate as integrate
from mpmath import whitw as whitw
from scipy import special
from scipy.constants import pi
from scipy.stats import norm
from examples.DataSaver import DataSaver


class Model():

    def __init__(self, risk_free_rate: float = 0, delta: float = 1 / 252, parameters: Optional[Dict[str, float]] = None,
                 lower_bound: float = 1e-10, upper_bound: float = 1e3):
        self.parameters = parameters
        self.risk_neutral_parameters = parameters
        self.delta = delta
        self.risk_free_rate = risk_free_rate
        self.cgf_bounds = [()]
        self.bounds = [(lower_bound,upper_bound)]


    def parameters_convention_update(self):
        pass

    def theoretical_cumulants(self) -> Dict:
        raise NotImplementedError

    def pdf(self, pdf_input: float, t: float):
        raise NotImplementedError

    def cdf_whittaker(self, cdf_input: float, t: float):
        raise NotImplementedError

    def cumulant(self, n:int):
        raise NotImplementedError

    def cumulant_generating_function(self, cgf_input: float):
        raise NotImplementedError

    def risk_neutral_parameters_update(self, p_star: float) -> Dict:
        raise NotImplementedError

    def chf(self, chf_input: complex, t: float, risk_neutral: bool) -> complex:
        raise NotImplementedError

    def risk_neutral_drift(self, chf_input: complex, t: float) -> complex:
        return np.exp(self.risk_free_rate * 1.j * chf_input * t)

    def parameters_update(self, parameters: np.ndarray) -> None:
        raise NotImplementedError

    def get_parameters(self):
        return self.parameters

    def update_cgf_bounds(self):
        pass

    def exponential_convexity_correction(self) -> float:
        pass

    def p_star_minimizer(self, cgf_input: float) -> float:
        pass

    def absolute_p_star_minimizer(self, cgf_input: float) -> float:
        pass


    def cdf(self, cdf_input: float, t: float, limit: int = int(1e9), upper_bound: float = 1e2,
                 singularity_tolerance: float = 1e-14) -> float:
        try:
            integral = integrate.quad(lambda u: np.real(
                (np.exp(-1j * u * cdf_input) * self.chf(chf_input=u, t=t, risk_neutral=False) / (1j * u))),
                                      a=singularity_tolerance, b=upper_bound, limit=limit)[0]
            return 0.5 - (1 / pi) * integral
        except Exception as e_ifft:
            DataSaver.log(text=f"Error calculating default_probability_ifft: {e_ifft}", console=True)
            return -1

    def check_bounds(self):
        combined_items = chain(self.parameters.items(), self.risk_neutral_parameters.items())
        for (param_key, param_value), (lower_bound, _) in zip(combined_items, chain(self.bounds, self.bounds)):
            if param_value <= lower_bound:
                raise ValueError(f"the value of {param_key} is {param_value} | constraint not respected")

    def theoretical_cumulants_update(self, parameters: np.ndarray) -> Dict:
        self.parameters_update(parameters=parameters)
        return self.theoretical_cumulants()

    def exponential_levy_chf(self, chf_input: complex, t: float) -> complex:
        return self.risk_neutral_drift(chf_input=chf_input, t=t) * self.chf(chf_input=chf_input, t=t,
                                                                            risk_neutral=True)


class Merton(Model):

    def __init__(self, risk_free_rate: float = 0, delta: float = 1 / 252, parameters: Optional[Dict[str, float]] = None,
                 lower_bound: float = 1e-8, upper_bound: float = 4.0,
                 ):
        super().__init__(risk_free_rate=risk_free_rate, delta=delta, parameters=parameters, lower_bound=lower_bound, upper_bound=upper_bound
                         )

        if parameters is None:
            parameters = {"sigma": 1e-2}

        # physical parameters dictionary
        self.parameters = parameters

        # risk-neutral parameters dictionary
        self.risk_neutral_parameters = self.risk_neutral_parameters_update(p_star=0)



    @property
    def sigma(self):
        return self.parameters["sigma"]

    @sigma.setter
    def sigma(self, value: float):
        self.parameters["sigma"] = value

    # second central moment, see chapter 2.5.1
    def second_central_moment(self) -> float:
        return self.sigma ** 2 * self.delta

    def theoretical_cumulants(self) -> Dict:
        return {
            'variance': self.second_central_moment(),
        }

    def parameters_update(self, parameters: np.ndarray) -> None:
        self.sigma = parameters[0]

    def cumulant_generating_function(self, cgf_input: float) -> float:
        return 0.5 * (self.sigma ** 2) * (cgf_input ** 2) * self.delta

    def risk_neutral_parameters_update(self, p_star: float) -> Dict:
        return {"sigma": self.sigma}

    def original_chf(self, chf_input: complex, t: float) -> complex:
        return np.exp(-0.5 * (self.sigma ** 2) * (chf_input ** 2) * t)

    # characteristic function of the bilateral gamma process
    def chf(self, chf_input: complex, t: float, risk_neutral: bool) -> complex:
        if risk_neutral:
            p_star = -0.5
            return (self.original_chf(chf_input=chf_input - 1.j * p_star, t=t)) / (
                self.original_chf(chf_input=-1.j * p_star, t=t))
        else:
            return self.original_chf(chf_input=chf_input, t=t)


    def cdf(self, cdf_input: float, t: float, limit: int = int(1e9), upper_bound: float = 1e2,
                 singularity_tolerance: float = 1e-14):
        std= self.sigma*np.sqrt(t)
        return norm.cdf(x=cdf_input, scale=std)

class BilateralGammaMotion(Model):
    def __init__(self, risk_free_rate: float = 0, delta: float = 1 / 252, parameters: Optional[Dict[str, float]] = None,
                 lower_bound: float = 1e-2, upper_bound: float = 1e3,
                 ):
        super().__init__(risk_free_rate=risk_free_rate, delta=delta, parameters=parameters,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound)

        if parameters is None:
            parameters = {"alpha_P": 1 + 1e-1, "lambda_P": 2 + 1e-1, "alpha_M": 1 + 1e-1, "lambda_M": 2 + 1e-1,
                          "sigma": 0.3}

        # physical parameters dictionary
        self.parameters = parameters

        # risk-neutral parameters dictionary
        self.risk_neutral_parameters = self.risk_neutral_parameters_update(p_star=0)

        self.cgf_bounds = [(
            -(self.lambda_M) + min(self.lambda_M, self.lambda_P) / 10,
            self.lambda_P - min(self.lambda_M, self.lambda_P) / 10
        )]


        self.bounds = [(lower_bound, upper_bound),
                                    (1 + lower_bound, upper_bound),
                                    (lower_bound, upper_bound),
                                    (lower_bound, upper_bound),
                                    (lower_bound/1e2, upper_bound/1e2)]


    @property
    def alpha_P(self):
        return self.parameters["alpha_P"]

    @alpha_P.setter
    def alpha_P(self, value: float):
        self.parameters["alpha_P"] = value

    @property
    def lambda_P(self):
        return self.parameters["lambda_P"]

    @lambda_P.setter
    def lambda_P(self, value: float):
        self.parameters["lambda_P"] = value

    @property
    def alpha_M(self):
        return self.parameters["alpha_M"]

    @alpha_M.setter
    def alpha_M(self, value: float):
        self.parameters["alpha_M"] = value

    @property
    def lambda_M(self):
        return self.parameters["lambda_M"]

    @lambda_M.setter
    def lambda_M(self, value: float):
        self.parameters["lambda_M"] = value

    @property
    def sigma(self):
        return self.parameters["sigma"]

    @sigma.setter
    def sigma(self, value: float):
        self.parameters["sigma"] = value


    def brownian_factor(self, n:int):
        return  self.sigma ** 2 if n == 2 else 0

    def cumulant(self, n: int):
        return math.factorial(n - 1) * (
                    self.brownian_factor(n=n) + self.alpha_P / (self.lambda_P ** n) + ((-1) ** n) * self.alpha_M / (
                        self.lambda_M ** n)) * self.delta
    
    

    def weekly_cumulant(self, n: int):
        return self.cumulant(n=n) * 7

    def adjusted_third_central_moment(self) -> float:
        return self.cumulant(n=3) / (self.cumulant(n=2) ** (1.5))

    def adjusted_fourth_cumulant(self) -> float:
        return self.fourth_central_moment() / (self.cumulant(n=2) ** (2))

    def fourth_central_moment(self) -> float:
        return self.cumulant(n=4) + 3 * (self.cumulant(n=2) ** 2)

    def fifth_central_moment(self) -> float:
        return self.cumulant(n=5) + 10 * self.cumulant(n=3) * self.cumulant(n=2)



    def theoretical_cumulants(self) -> Dict:
        return {'mean': self.cumulant(n=1),
            'variance': self.cumulant(n=2),
            'skewness': self.cumulant(n=3),
        'fourth_cumulant' : self.cumulant(n=4),
        'sixth_cumulant':self.cumulant(n=6)}




    def parameters_update(self, parameters: np.ndarray) -> None:
        self.alpha_P = parameters[0]
        self.lambda_P = parameters[1]
        self.alpha_M = parameters[2]
        self.lambda_M = parameters[3]
        self.sigma = parameters[4]

    # update and return theoretical moments
    def update_cgf_bounds(self):
        self.cgf_bounds = [(
            -(self.lambda_M) + 1e-1,
            self.lambda_P - 1e-1
        )]

    def bilateral_gamma_cgf(self, cgf_input: float) -> float:
        return (np.log(
            ((self.lambda_P / (self.lambda_P - cgf_input)) ** self.alpha_P)
            *
            ((self.lambda_M / (self.lambda_M + cgf_input)) ** self.alpha_M))
                ) * self.delta

    def cumulant_generating_function(self, cgf_input: float) -> float:
        return self.bilateral_gamma_cgf(cgf_input=cgf_input)+ (0.5 * (self.sigma ** 2) * cgf_input ** 2
                ) * self.delta

    def brownian_factor_p_star(self, cgf_input:float)->float:
        return np.exp(0.5 * (self.sigma ** 2) * (((cgf_input + 1) ** 2) - (cgf_input ** 2)))

    def p_star_minimizer(self, cgf_input: float) -> float:
        increased_cgf_input = cgf_input + 1
        eta = (((self.lambda_P / (self.lambda_P - increased_cgf_input)) ** self.alpha_P) *
               ((self.lambda_M / (self.lambda_M + increased_cgf_input)) ** self.alpha_M)) / \
              (((self.lambda_P / (self.lambda_P - cgf_input)) ** self.alpha_P) *
               ((self.lambda_M / (self.lambda_M + cgf_input)) ** self.alpha_M))
        brownian_factor = self.brownian_factor_p_star(cgf_input=cgf_input)
        return eta + brownian_factor - 1

    def absolute_p_star_minimizer(self, cgf_input: float) -> float:
        return np.abs(self.p_star_minimizer(cgf_input=cgf_input))

    def bilateral_gamma_risk_neutral_update(self,p_star:float)->Dict:
        return {"alpha_P_RN": self.alpha_P, "lambda_P_RN": self.lambda_P - p_star, "alpha_M_RN": self.alpha_M,
         "lambda_M_RN": self.lambda_M + p_star}
    def risk_neutral_parameters_update(self, p_star: float) -> Dict:
        risk_neutral_parameters= self.bilateral_gamma_risk_neutral_update(p_star=p_star)
        risk_neutral_parameters["sigma_RN"]= self.sigma
        self.mu = (self.sigma ** 2) * p_star
        return risk_neutral_parameters

    def risk_neutral_choice(self, risk_neutral: bool):
        if risk_neutral:
            return self.risk_neutral_parameters['alpha_P_RN'], \
                self.risk_neutral_parameters['lambda_P_RN'], self.risk_neutral_parameters['alpha_M_RN'], \
                self.risk_neutral_parameters['lambda_M_RN']
        else:
            return self.parameters['alpha_P'], \
                self.parameters['lambda_P'], self.parameters['alpha_M'], \
                self.parameters['lambda_M']


    def bilateral_gamma_chf(self, chf_input: complex, t: float, risk_neutral: bool):
        alpha_P, lambda_P, alpha_M, lambda_M = self.risk_neutral_choice(risk_neutral=risk_neutral)
        return ((lambda_P / (lambda_P - 1.j * chf_input)) ** (alpha_P * t)) \
            * ((lambda_M / (lambda_M + 1.j * chf_input)) ** (alpha_M * t))

    def chf(self, chf_input: complex, t: float, risk_neutral: bool) -> complex:
        bilateral_gamma_chf =self.bilateral_gamma_chf(chf_input=chf_input, t=t, risk_neutral=risk_neutral)
        mu = self.mu if risk_neutral else 0
        return np.exp((1.j * mu * chf_input * t) - 0.5 * (self.sigma ** 2) * (chf_input ** 2) * t) * bilateral_gamma_chf



class BilateralGamma(BilateralGammaMotion):

    def __init__(self, risk_free_rate: float = 0, delta: float = 1 / 252, parameters: Optional[Dict[str, float]] = None,
                 lower_bound: float = 1e-2, upper_bound: float = 1e3,
                 ):
        super(BilateralGammaMotion, self).__init__(risk_free_rate=risk_free_rate, delta=delta, parameters=parameters,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound)

        if parameters is None:
            parameters = {"alpha_P": 10 + 1e-1, "lambda_P": 20 + 1 + 1e-1, "alpha_M": 10 + 1e-1, "lambda_M": 20 + 1e-1}

        # physical parameters dictionary
        self.parameters = parameters

        # risk-neutral parameters dictionary
        self.risk_neutral_parameters = self.risk_neutral_parameters_update(p_star=0)



        self.bounds = [(lower_bound, upper_bound),
                                    (1 + lower_bound, upper_bound),
                                    (lower_bound, upper_bound),
                                    (lower_bound, upper_bound)]



    @staticmethod
    # whittaker function, used for pdf. See appendix A
    def whittaker_function(pdf_input: float, t: float, alpha_P: float, lambda_P: float, alpha_M: float,
                           lambda_M: float) -> float:
        lambda_whittaker = 0.5 * (alpha_P - alpha_M) * t
        mu = 0.5 * ((alpha_P + alpha_M) * t - 1)
        whitt_input = (lambda_P + lambda_M) * pdf_input
        return float(whitw(lambda_whittaker, mu, whitt_input).real)

    @staticmethod
    # factor computation for pdf, see chapter 2.1
    def pdf_sqrt_adjustment(pdf_input: float, t: float, alpha_P: float, lambda_P: float, alpha_M: float,
                            lambda_M: float) -> float:
        whittaker_value = BilateralGamma.whittaker_function(pdf_input=pdf_input, t=t, alpha_P=alpha_P,
                                                            lambda_P=lambda_P, alpha_M=alpha_M,
                                                            lambda_M=lambda_M)
        if whittaker_value == float(0):
            return 0
        lambda_factor = (lambda_P ** (alpha_P * t)) * (lambda_M ** (alpha_M * t))
        pdf_input_factor = np.sqrt(whittaker_value) * (pdf_input ** (0.5 * (alpha_P + alpha_M) * t - 1)) / (
                (lambda_P + lambda_M) ** (0.5 * (alpha_P + alpha_M) * t))
        exponential_factor = np.sqrt(whittaker_value) * (
            np.exp(- 0.5 * (lambda_P - lambda_M) * pdf_input)) / special.gamma(alpha_P * t)
        pdf_value = lambda_factor * pdf_input_factor * exponential_factor
        if not isinstance(pdf_value, float) or math.isnan(pdf_value):
            return 0.0
        return pdf_value



    # adjustment of density function's parameters, necessary for the Bilateral Gamma process
    def pdf_parameters_adjustment(self, pdf_input: float) -> Tuple[float, float, float, float, float]:
        return (abs(pdf_input), self.alpha_M, self.lambda_M, self.alpha_P, self.lambda_P) if pdf_input <= 0 else (
            pdf_input, self.alpha_P, self.lambda_P, self.alpha_M, self.lambda_M)

    # pdf, see chapter 2.1
    def pdf(self, pdf_input: float, t: float) -> float:
        pdf_input, alpha_P, lambda_P, alpha_M, lambda_M = self.pdf_parameters_adjustment(pdf_input=pdf_input)
        return BilateralGamma.pdf_sqrt_adjustment(pdf_input=pdf_input, t=t, alpha_P=alpha_P, lambda_P=lambda_P,
                                                  alpha_M=alpha_M,
                                                  lambda_M=lambda_M)

    # cdf of the Bilateral Gamma process
    def cdf_whittaker(self, cdf_input: float, t: float) -> float:
        # TODO : Handle numerical instabilities
        try:
            return integrate.quad(lambda x: self.pdf(x, t), -100, cdf_input)[0]
        except Exception as e_whitt:
            raise ValueError("Error while computing cdf using Whittaker")

    def brownian_factor_cumulants(self):
        return 0
    def brownian_factor_p_star(self, cgf_input:float):
        return 0

    def theoretical_cumulants(self) -> Dict:
        theoretical_cumulants = super().theoretical_cumulants()
        theoretical_cumulants.pop("sixth_cumulant")
        return theoretical_cumulants

    def parameters_update(self, parameters: np.ndarray) -> None:
        self.alpha_P = parameters[0]
        self.lambda_P = parameters[1]
        self.alpha_M = parameters[2]
        self.lambda_M = parameters[3]

    # update and return theoretical moments

    def cumulant_generating_function(self, cgf_input: float) -> float:
        return self.bilateral_gamma_cgf(cgf_input=cgf_input)

    def constraints(self, cgf_input: np.ndarray):
        cgf_input = cgf_input[0]
        return ((self.lambda_P / (self.lambda_P - cgf_input)) ** self.alpha_P) * (
                (self.lambda_M / (self.lambda_M + cgf_input)) ** self.alpha_M)

    def risk_neutral_parameters_update(self, p_star: float) -> Dict:
        return self.bilateral_gamma_risk_neutral_update(p_star=p_star)

    def chf(self, chf_input: complex, t: float, risk_neutral: bool) -> complex:
        return self.bilateral_gamma_chf(chf_input=chf_input, t=t, risk_neutral=risk_neutral)

    # computation of exponential of the convexity correction, in order to avoid the computation
    # of both exponential and log
    def exponential_convexity_correction(self) -> float:
        alpha_P_RN, lambda_P_RN, alpha_M_RN, lambda_M_RN = self.risk_neutral_choice(risk_neutral=False)
        return (((lambda_P_RN / (lambda_P_RN - 1)) ** alpha_P_RN) * (
                (lambda_M_RN / (lambda_M_RN + 1)) ** alpha_M_RN))




    # characteristic function of the exponential Bilateral Gamma process





class VarianceGamma(BilateralGamma):

    def __init__(self, risk_free_rate: float = 0, delta: float = 1 / 252, parameters: Optional[Dict[str, float]] = None,
                 lower_bound: float = 1e-2, upper_bound: float = 1e3):
        super(BilateralGammaMotion,self).__init__(risk_free_rate=risk_free_rate, delta=delta, parameters=parameters)

        if parameters is None:
            parameters = {"alpha": 10 + 1e-1, "lambda_P": 20 + 1 + 1e-1, "lambda_M": 20 + 1e-1}

        # physical parameters dictionary
        self.parameters = parameters

        # risk-neutral parameters dictionary
        self.risk_neutral_parameters = self.risk_neutral_parameters_update(p_star=0)


        self.bounds = [(lower_bound, upper_bound),
                                    (1 + lower_bound, upper_bound),
                                    (lower_bound, upper_bound)]



    @property
    def alpha(self):
        return self.parameters["alpha"]

    @alpha.setter
    def alpha(self, value: float):
        self.parameters["alpha"] = value



    def cumulant(self, n: int):
        return math.factorial(n - 1) * (
                    self.alpha / (self.lambda_P ** n) + ((-1) ** n) * self.alpha / (self.lambda_M ** n)) * self.delta


    def theoretical_cumulants(self) -> Dict:
        theoretical_cumulants = super().theoretical_cumulants()
        theoretical_cumulants.pop("skewness")

        return theoretical_cumulants


    def parameters_update(self, parameters: np.ndarray) -> None:
        self.alpha = parameters[0]
        self.lambda_P = parameters[1]
        self.lambda_M = parameters[2]

    # update and return theoretical moments

    def cumulant_generating_function(self, cgf_input: float) -> float:
        return (np.log(
            ((self.lambda_P / (self.lambda_P - cgf_input)) ** self.alpha)
            *
            ((self.lambda_M / (self.lambda_M + cgf_input)) ** self.alpha))
        ) * self.delta

    def constraints(self, cgf_input: np.ndarray):
        cgf_input = cgf_input[0]
        return ((self.lambda_P / (self.lambda_P - cgf_input)) ** self.alpha) * (
                (self.lambda_M / (self.lambda_M + cgf_input)) ** self.alpha)

    def risk_neutral_parameters_update(self, p_star: float) -> Dict:
        return {"alpha_RN": self.alpha, "lambda_P_RN": self.lambda_P - p_star,
                "lambda_M_RN": self.lambda_M + p_star}

    def risk_neutral_choice(self, risk_neutral: bool):
        if risk_neutral:
            return self.risk_neutral_parameters['alpha_RN'], \
                self.risk_neutral_parameters['lambda_P_RN'], \
                self.risk_neutral_parameters['lambda_M_RN']
        else:
            return self.parameters['alpha'], \
                self.parameters['lambda_P'], \
                self.parameters['lambda_M']

    # computation of exponential of the convexity correction, in order to avoid the computation
    # of both exponential and log
    def exponential_convexity_correction(self) -> float:
        alpha_RN, lambda_P_RN, lambda_M_RN = self.risk_neutral_choice(risk_neutral=False)
        return (((lambda_P_RN / (lambda_P_RN - 1)) ** alpha_RN) * (
                (lambda_M_RN / (lambda_M_RN + 1)) ** alpha_RN))



    # characteristic function of the bilateral gamma process
    def chf(self, chf_input: complex, t: float, risk_neutral: bool) -> complex:
        alpha, lambda_P, lambda_M = self.risk_neutral_choice(risk_neutral=risk_neutral)
        return ((lambda_P / (lambda_P - 1.j * chf_input)) ** (alpha * t)) \
            * ((lambda_M / (lambda_M + 1.j * chf_input)) ** (alpha * t))

    def p_star_minimizer(self, cgf_input: float) -> float:
        increased_cgf_input = cgf_input + 1
        eta = (((self.lambda_P / (self.lambda_P - increased_cgf_input)) ** self.alpha) *
               ((self.lambda_M / (self.lambda_M + increased_cgf_input)) ** self.alpha)) / \
              (((self.lambda_P / (self.lambda_P - cgf_input)) ** self.alpha) *
               ((self.lambda_M / (self.lambda_M + cgf_input)) ** self.alpha))
        return eta - 1

    def absolute_p_star_minimizer(self, cgf_input: float) -> float:
        return np.abs(self.p_star_minimizer(cgf_input=cgf_input))

    def parameters_convention_update(self):
        alpha = self.alpha
        lambda_P = self.lambda_P
        lambda_M = self.lambda_M
        self.parameters["sigma"] = np.sqrt((2 * alpha) / (lambda_P * lambda_M))
        self.parameters["theta"] = alpha / lambda_P - alpha / lambda_M
        self.parameters["nu"] = 1 / alpha

