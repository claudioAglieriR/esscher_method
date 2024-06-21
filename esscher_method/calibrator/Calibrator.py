import copy
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
from typing import Dict, List, Tuple, Callable, Union

import numpy as np
from scipy import optimize
from scipy.stats import moment

from esscher_method.model.Model import Model
from examples.DataSaver import DataSaver
from .LewisEuropeanTargetPricer import LewisEuropeanTargetPricer


class Calibrator:

    def __init__(self, equity_values_list,
                 model: Model = Model(),
                 debt: float = 0,
                 maturity: float = 1,
                 days_number: int = 252,
                 delta: float = 1 / 252,
                 minimization_diff_evolution: bool = True,
                 tolerance: float = 1e-7,
                 max_iteration: int = 20, verbose: int = 1
                 ):
        """
        :param maturity horizon used to compute default probability, expressed in years (e.g. 1 month=1/12)
        :param minimization_diff_evolution: if True, the differential evolution algorithm is used to retrieve model parameters
        :param tolerance: threshold required to quit the calibration procedure
        :param verbose: similar to scipy standard:    0 : work silently.
                                                      1 (default): save all the results in the log file, and display main results on console.
                                                      2 : save all the results in the log file, and display all results on console
        """

        self.model = model
        self.model.delta = delta
        self.pricer = LewisEuropeanTargetPricer(model=model, K=debt)

        self.debt = debt
        self.equity_values_list = equity_values_list

        self.maturity = maturity
        self.days_number = days_number

        self.sample_cumulants = {'mean': 0.0, 'variance': 0.0, 'skewness': 0.0}

        self.physical_history = []
        self.risk_neutral_history = []

        self.iterations_number = 0

        self.minimization_diff_evolution = minimization_diff_evolution

        self.tolerance = tolerance
        self.max_iteration = max_iteration

        self.p_star = 0
        self.verbose = verbose
        self.console = True if verbose == 2 else False

    def differential_evolution(self, objective_function: Callable, bounds: List[Tuple[float, float]],
                               strategy: str = 'best1bin',
                               maxiter: int = 10 ** 9,
                               tol: float = 1e-9, mutation: Tuple = (0.5, 1.5), recombination: float = 0.95,
                               polish: bool = True, init: str = 'latinhypercube', updating: str = 'immediate',
                               popsize: int = 50):
        return optimize.differential_evolution(
            objective_function,
            bounds=bounds,
            strategy=strategy,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            polish=polish,
            init=init,
            updating=updating
        ).x

    @staticmethod
    def sixth_unbiased_cumulant_estimate(n: int, m_2: float, m_3: float, m_4: float, m_6: float) -> np.ndarray:

        den_sixth_cumulant = (n - 5) * (n - 4) * (n - 3) * (n - 2) * (n - 1) * n
        second_moment_correction = n ** 3 * (60 * n - 90 * (n ** 2) + 30 * (n ** 3)) * (m_2 ** 3)
        third_moment_correction = n ** 2 * (40 * n - 50 * (n ** 2) + 20 * (n ** 3) - 10 * (n ** 4)) * (m_3 ** 2)
        multivariate_correction = n ** 2 * (-60 * n + 105 * (n ** 2) - 30 * (n ** 3) - 15 * (n ** 4)) * (m_2 * m_4)
        sixth_moment_correction = n * (-4 * (n ** 2) + 11 * (n ** 3) + 16 * (n ** 4) + n ** 5) * (m_6)
        return (second_moment_correction + third_moment_correction + multivariate_correction + sixth_moment_correction)/ den_sixth_cumulant

    def fifth_unbiased_cumulant_estimate(self, n: int, m_2: float, m_3: float, m_5: float):
        M_5_unbiased = (((n ** 2) * (20 * n - 10 * (n ** 2)) * m_2 * m_3) + (
                n * (10 * (n ** 2) - 5 * (n ** 3) + (n ** 4)) * m_5)) / (
                               (n - 4) * (n - 3) * (n - 2) * (n - 1) * n)
        self.sample_cumulants['fifth_cumulant'] = M_5_unbiased - 10 * self.sample_cumulants['skewness'] * \
                                                  self.sample_cumulants['variance']

    def adjusted_cumulants_estimates(self, n: int, m_2: float, m_3: float, m_4: float):
        g_1 = m_3 / (m_2 ** (1.5))
        g_2 = m_4 / (m_2 ** 2) - 3
        self.sample_cumulants['skewness'] = (np.sqrt(n * (n - 1))) / (n - 2) * g_1
        self.sample_cumulants['kurtosis'] = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g_2 + 6)

    def unbiased_central_moments_estimates(self, n: int, m_2: float, m_3: float, m_4: float, m_5: float):
        self.sample_cumulants['kurtosis'] = (((9 - 6 * n) * (n ** 2) * (m_2 ** 2)) + (
                    n * ((3 * n) - 2 * (n ** 2) + (n ** 3)) * m_4)) / ((n - 3) * (n - 2) * (n - 1) * n)
        self.sample_cumulants['hyper_skewness'] = (((n ** 2) * (20 * n - 10 * (n ** 2)) * m_2 * m_3) + (
                    n * (10 * (n ** 2) - 5 * (n ** 3) + (n ** 4)) * m_5)) / ((n - 4) * (n - 3) * (n - 2) * (n - 1) * n)


    def sample_moments_update(self, data: np.ndarray) -> None:
        """
        method used to set the first five moments in the Calibrator object

        :param data: assets values
        """

        m_2 = moment(data, moment=2)
        n = len(data)
        self.sample_cumulants['variance'] = n / (n - 1) * m_2
        if len(self.model.parameters) > 1:
            m_3 = moment(data, moment=3)
            m_4 = moment(data, moment=4)
            m_6 = moment(data, moment=6)
            self.sample_cumulants['mean'] = mean(data)
            self.sample_cumulants['skewness'] = n ** 2 / ((n - 1) * (n - 2)) * m_3
            self.sample_cumulants['fourth_cumulant'] = n ** 2 / ((n - 1) * (n - 2) * (n - 3)) * (
                   (n + 1) * m_4 - 3 * (n - 1) * m_2 ** 2)

            self.sample_cumulants['sixth_cumulant'] = Calibrator.sixth_unbiased_cumulant_estimate(n=n, m_2=m_2, m_3=m_3,
                                                                                                 m_4=m_4, m_6=m_6)




    def objective_function(self, parameters: np.ndarray) -> float:
        """
        objective function used to compute residuals for differential evolution

        :param parameters: numpy array containing the parameters values for a chosen model
        :return: sum of the relative residuals
        """
        theoretical_cumulants = self.model.theoretical_cumulants_update(parameters=parameters)
        sum_relative_errors = 0
        for key in theoretical_cumulants:
            error = abs((theoretical_cumulants[key] - self.sample_cumulants[key]) / self.sample_cumulants[key])
            sum_relative_errors += error
        return sum_relative_errors

    def moments_residuals(self, parameters: np.ndarray, relative: bool = False) -> np.ndarray:
        """
        method used to monitor moments residuals

        :param parameters: numpy array containing the parameters values for a chosen model
        :return: residuals for each moment
        """
        theoretical_cumulants = self.model.theoretical_cumulants_update(parameters=parameters)
        if not relative:
            return np.array([theoretical_cumulants[key] - self.sample_cumulants[key] for key in theoretical_cumulants])
        else:
            return np.array([
                abs((theoretical_cumulants[key] - self.sample_cumulants[key]) / abs(self.sample_cumulants[key])) * 100 for
                key in theoretical_cumulants])




    def esscher_equation_residual(self, cgf_input: float) -> float:
        """
        method to compute esscher equation residuals, in order to find the optimal cgf_input
        in step 2 (see Algorithm 1 description)

        :param cgf_input: input of the cumulant generating function
        :return: residual of the esscher equation
        """
        return self.model.cumulant_generating_function(
            cgf_input=cgf_input + 1) - self.model.cumulant_generating_function(cgf_input=cgf_input)

    def theoretical_parameters_computation(self, initial_guess: np.ndarray) -> Dict:
        """
        method used to compute the optimal model parameters, given the sample parameters
        if self.minimization_diff_evolution is True, differential evolution algorithm is used

        :param initial_guess: array given as starting point for the minimize function
        :return: optimal parameters as a dictionary, where the keys are the names of the parameters
        """
        if self.minimization_diff_evolution and len(self.model.parameters) > 1:
            parameters = self.differential_evolution(objective_function=self.objective_function,
                                                     bounds=self.model.bounds)
        else:
            parameters = optimize.fsolve(func=self.moments_residuals, x0=initial_guess)

        return {param_name: param_value for param_name, param_value in
                zip(list(self.model.parameters.keys()), parameters)}

    def check_model_update(self) -> None:
        if self.model != self.pricer.model:
            raise ValueError("The model is not correctly updated")

    def model_pricer_parameters_update(self, parameters: Dict, risk_neutral: bool) -> None:
        """
        method used to update the list of parameters and model's parameters

        :param parameters: updated parameters
        :param risk_neutral: boolean, if true the risk neutral parameters are updated
        """
        if risk_neutral:
            self.risk_neutral_history.append(parameters.copy())
            self.model.risk_neutral_parameters = parameters.copy()
        else:
            self.physical_history.append(parameters.copy())
            self.model.parameters = parameters.copy()

            if self.verbose != 0:
                DataSaver.physical_monitoring(calibrator=self, parameters=np.array(list(self.physical_history[-1].values())))


        self.model.check_bounds()
        self.check_model_update()

    def physical_update(self, data: np.ndarray, initial_guess: np.ndarray) -> None:
        """
        update of physical parameters, necessary during step 1 and step 4 (see Algorithm 1 description)

        :param data: assets values
        :param initial_guess: starting point for minimization (if differential evolution is not used)
        """
        self.sample_moments_update(data=data)
        parameters = self.theoretical_parameters_computation(initial_guess=initial_guess)
        self.model_pricer_parameters_update(parameters=parameters, risk_neutral=False)

    def pre_differential_evolution_init(self, data: np.ndarray) -> np.ndarray:
        """
        STEP 0 METHOD
        If differential evolution method is not used, this method is used to find an appropriate initial guess.

        :param data: equity values
        :return: optimal parameters of the model
        """
        # Update of sample moments (M_e) using equity_returns
        self.sample_moments_update(data=data)

        parameters = self.differential_evolution(objective_function=self.objective_function,
                                                 bounds=self.model.bounds)

        DataSaver.physical_monitoring(calibrator=self,parameters=parameters)

        return parameters

    def historical_values_init(self) -> None:
        """
        STEP 1 (see Algorithm 1 description)
        """

        DataSaver.log("\n\nSTEP 1 - historical_values_init")
        self.physical_history.append(self.model.parameters.copy())
        self.risk_neutral_history.append(self.model.risk_neutral_parameters.copy())

        # Computation of returns using equity values
        equity_returns = np.diff(np.log(self.equity_values_list), n=1)

        if not self.minimization_diff_evolution and len(self.model.parameters) > 1:
            # First parameters initialization with Differential Evolution
            # NOTE: This additional initialization has been performed in other to provide an initial guess for the fsolve method
            initial_guess = self.pre_differential_evolution_init(data=equity_returns)
        else:
            initial_guess = np.array(list(self.model.parameters.values()))

        # Update of physical parameters
        self.physical_update(initial_guess=initial_guess, data=equity_returns)
        self.risk_neutral_update()

    def risk_neutral_update(self) -> None:
        """
        STEP 2 (see Algorithm 1 description)
        """

        self.model.update_cgf_bounds()
        # obtain the optimal p, i.e. the optimal input for the cumulant generating function

        self.p_star = float(
            optimize.fsolve(func=self.esscher_equation_residual, x0=np.array([self.p_star]))[0])
        parameters = self.model.risk_neutral_parameters_update(p_star=self.p_star)
        self.model_pricer_parameters_update(parameters=parameters, risk_neutral=True)
        DataSaver.log(text=f"p_star FSOLVE= {self.p_star}", console=self.console)

    def final_asset_residual_computation(self):
        self.pricer.T = self.maturity
        self.pricer.target_price = self.equity_values_list[-1]
        self.final_asset_residual = self.pricer.price_residual(S0=self.asset_values[-1])
        DataSaver.log(text=f"\nFinal asset residual: \n{self.final_asset_residual} \n", console=self.console)

    def daily_asset_minimization(self, day):

        dedicated_pricer = copy.deepcopy(self.pricer)
        dedicated_pricer.T = self.maturity * (1 + (self.days_number - day - 1) / self.days_number)
        dedicated_pricer.target_price = self.equity_values_list[day]
        return float(optimize.minimize(
            fun=dedicated_pricer.absolute_price_residual,
            x0=np.array([dedicated_pricer.target_price + dedicated_pricer.K]),
            bounds=[(dedicated_pricer.target_price, np.iinfo(np.int64).max)], method='Nelder-Mead').x)

    def asset_log_returns_computation(self):
        num_days = self.equity_values_list.shape[0]
        DataSaver.log(
            text=f"\nRisk Neutral Parameters used for pricing: \n{self.pricer.model.risk_neutral_parameters} \n",
            console=self.console)

        try:
            with ProcessPoolExecutor() as executor:
                asset_values = list(executor.map(self.daily_asset_minimization, range(num_days)))

            self.asset_values = asset_values
            DataSaver.log(text=f"\nASSET VALUES: \n{asset_values} \n", console=self.console)
            self.final_asset_residual_computation()
            return np.diff(np.log(asset_values), n=1)

        except Exception as e:
            raise ValueError(f"Error asset_log_returns_computation: {e}")

    def recurrent_estimation(self) -> None:
        """
        ITERATIVE STEPS 3 TO 5 (see Algorithm 1 description)
        """
        continue_loop = True
        while continue_loop:

            # LOOP CONDITION : SEE STEP 5
            # Reset the condition to False. It will be set to True if at least one difference condition exceeds the tolerance
            continue_loop = False

            # Get the keys from the last dictionary in physical_history (assuming all dictionaries have the same keys)
            keys = self.physical_history[-1].keys()

            # Check each key for the tolerance condition
            for key in keys:
                if np.abs(self.physical_history[-1][key] - self.physical_history[-2][
                    key]) > self.tolerance and self.iterations_number < self.max_iteration:
                    continue_loop = True
                    break  # Break the for loop if a condition is true

            self.iterations_number += 1

            # STEP 2
            DataSaver.log(text="STEP 2 - risk_neutral_update")
            self.risk_neutral_update()

            # STEP 3
            DataSaver.log(text="STEP 3 - asset_log_returns_computation")
            try:
                asset_returns = np.array(self.asset_log_returns_computation())
                DataSaver.log(f"Asset Returns elements: {len(asset_returns)}",
                                    console=self.console)
            except Exception as e:
                raise ValueError(f"Error while computing assets log returns: {e}")
                # break
            # STEP 4
            DataSaver.log(text="STEP 4 - physical_update")
            self.physical_update(initial_guess=np.array(list(self.physical_history[-1].values())), data=asset_returns)

    def default_probability_computation(self) -> Union[float,Tuple]:
        return self.model.cdf(cdf_input=-self.distance_to_default, t=self.maturity) * 100
    def final_residuals(self):

        residuals = {}

        residuals['cumulants'] = {cumulant_name: cumulant_residual for cumulant_name, cumulant_residual in
                                  zip(['first_cumulant_residual', 'second_cumulant_residual', 'third_cumulant_residual',
                                       'fourth_cumulant_residual', 'fifth_cumulant_residual'],
                                      self.moments_residuals(
                                          parameters=np.array(list(self.model.get_parameters().values())),
                                          relative=True))}

        residuals['final_asset_residual'] = self.final_asset_residual
        return residuals

    def calibration(self) :

        # STEP 1
        self.historical_values_init()

        # STEPS 2 TO 5
        self.recurrent_estimation()

        self.distance_to_default = np.log(self.asset_values[-1] / self.debt)


        DataSaver.results_monitoring(results={
            'default_probability': self.default_probability_computation(),
            'parameters': self.model.parameters,
            'residuals': self.final_residuals(),
            'iterations_number': self.iterations_number
        })


    def get_final_asset(self) -> float:
        return self.asset_values[-1]

    def get_number_iteration(self) -> int:
        return self.iterations_number

