import copy
import os
from typing import List, Tuple

import pandas as pd

from DataSaver import DataSaver
from esscher_method.calibrator.Calibrator import Calibrator
from esscher_method.model.Model import BilateralGamma, BilateralGammaMotion, VarianceGamma, Merton


class CalibrationExecutor:
    @staticmethod
    def execute_calibration(model_name: str, delta: float, tickers: List[str], time_spans: List[Tuple],
                            maturities: List[float], data_path:str,  upper_bound: float = 1e3,
                            verbose: int=2 ) -> Calibrator:
        """Main method to execute the calibration process."""


        models = {'BilateralGamma': BilateralGamma(delta=delta,
                                                   ),
                  'BilateralGammaMotion': BilateralGammaMotion(delta=delta,
                                                               ),
                  'VarianceGamma': VarianceGamma(delta=delta,
                                                 ),
                  'Merton': Merton(delta=delta,
                                   )
                  }

        for time_span, maturity in zip(time_spans, maturities):
            equity_data, debt_data = DataSaver.read_and_process_data(data_path, time_span=time_span)
            if equity_data is None or debt_data is None:
                raise ValueError("Equity data or debt data is missing.")

            model = models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} is missing.")

            for ticker in tickers:
                DataSaver.log(
                    f"Processing {ticker} | model = {model_name} | upper_bound = {upper_bound} | time_span = {time_span} | "
                    f"maturity = {maturity}")

                tolerance = 1e-5 if model_name == 'Merton' else 1e-3
                calibrator = CalibrationExecutor.create_calibrator(
                    ticker=ticker, equity_data=equity_data, debt_data=debt_data, tolerance=tolerance,
                    model=copy.deepcopy(model), delta=delta, maturity=maturity, verbose=verbose,
                )
                DataSaver.log(f"Days_number: {calibrator.days_number}", console=calibrator.console)
                DataSaver.log(f"Equity_Value_List: {len(calibrator.equity_values_list)}", console=calibrator.console)

                calibrator.calibration()
                calibrator.model.parameters_convention_update()

        return calibrator

    @staticmethod
    def create_calibrator(equity_data: pd.Series, debt_data: pd.Series, model,
                          ticker: str, delta: float, maturity: float,
                          tolerance: float = 1e-3,
                          verbose: int = 1) -> Calibrator:
        company_equity_ticker = f"{ticker} Equity"

        try:
            debt = debt_data[[company_equity_ticker]].iloc[0, 0]
            equity_values = equity_data[[company_equity_ticker]].iloc[:, 0]
        except KeyError as e:
            raise KeyError(f"Ticker {ticker} not found") from e

        return Calibrator(
            debt=debt,
            equity_values_list=equity_values,
            model=model,
            tolerance=tolerance,
            delta=delta,
            maturity=maturity,
            days_number=len(equity_values),
            minimization_diff_evolution=True,
            verbose=verbose
        )
