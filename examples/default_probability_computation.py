import os
from Calibration_Executor import CalibrationExecutor
from DataSaver import DataSaver



if __name__ == "__main__":

    log_file_path = os.path.dirname(os.path.abspath(__file__))
    file_name=input("Please enter the excel file name in the \'data folder\' (without extension): ")
    data_path = os.path.join('data', f'{file_name}.xlsx')
    models_names = ['Merton', 'VarianceGamma','BilateralGamma']


    data_single_maturity = {
              'tickers': ['SAP GY', 'DAI GY'],
        'time_spans':  [ ('2019-10-25', '2020-10-13')],
        'maturities': [1],
        'mode': 'single_maturity'}


    DataSaver.log_file_creation(log_file_path=log_file_path)
    for model_name in models_names:
        calibrator = CalibrationExecutor.execute_calibration(model_name=model_name,delta=1 / 252,
                                                             tickers=data_single_maturity['tickers'],
                                                                 time_spans=data_single_maturity['time_spans'],
                                                                 maturities=data_single_maturity['maturities'],
                                                              data_path=data_path,
                                                                  verbose=2)
    print(calibrator.default_probability_computation(),
          calibrator.model.parameters,
          calibrator.final_residuals(),
          calibrator.iterations_number)

