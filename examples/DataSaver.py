import datetime
import logging
import os
from typing import Tuple, Union, List, Dict
import numpy as np
import pandas as pd

class DataSaver:

    @staticmethod
    def read_and_process_data(file_path: str, time_span: Tuple[str, str]):
        try:
            data = pd.read_excel(file_path, sheet_name=None)
        except FileNotFoundError:
            print(f"Errore: File {file_path} non trovato.")
            return None, None

        data_equity = data['Mod Market Cap'].set_index('Dates').loc[time_span[0]:time_span[1]]

        data_debt = data['Gross Debt'].dropna()
        return data_equity, data_debt


    @staticmethod
    def update_parameters(keys: Union[List,np.ndarray], source_dict: Dict, data_dict: Dict):
        for key in keys:
            if key in source_dict:
                data_dict[key] = source_dict.get(key, 0.0)



    @staticmethod
    def create_log_file_name(base_file_name:str):
        index = 0
        base_name, extension = os.path.splitext(base_file_name)
        current_file_name = base_file_name
        while os.path.isfile(current_file_name):
            current_file_name = f"{base_name}_{index}{extension}"
            index += 1
        return current_file_name



    @staticmethod
    def results_monitoring(results:Dict, indent=0):
        max_key_length = max(len(key) for key in results.keys())
        for key, value in results.items():
            if isinstance(value, dict):
                DataSaver.log(" " * indent + f"{key}:")
                DataSaver.results_monitoring(value, indent=indent + 4)
            elif key == 'default_probability' and isinstance(value, tuple):
                formatted_tuple = ", ".join([f"{v:.4f}" for v in value])
                DataSaver.log(" " * indent + f"{key:>{max_key_length}} : ({formatted_tuple})")
            else:
                DataSaver.log(" " * indent + f"{key:>{max_key_length}} : {value:.4f}")
        DataSaver.log("\n")


    @staticmethod
    def log(text: str, console: bool=True) -> None:
        logging.info(text)
        if console:
            DataSaver.print_with_time(text)

    @staticmethod
    def print_with_time(message: str) -> None:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{current_time} - {message}")

    @staticmethod
    def format_moments(moments_dict: Dict):
        return " | ".join(f"{moment.capitalize()}: {value:.10f}" for moment, value in moments_dict.items())

    @staticmethod
    def format_errors(differences:Union[List,np.ndarray], moments_dict:Dict, error_type:str='Relative'):
        errors = []
        num_iterations = min(len(differences), len(moments_dict))

        for i, (moment, value) in enumerate(list(moments_dict.items())[:num_iterations]):
            if value == 0:
                error_message = "undefined (division by zero)"
            else:
                error_value = differences[i] / value if error_type == 'Relative' else differences[i]
                error_message = f"{error_value}"
            errors.append(f"{error_type} Errors: {i + 1} Moment {error_message}")
        return " | ".join(errors)

    @staticmethod
    def log_file_creation(log_file_path: str) :
        try:
            os.makedirs(log_file_path)
        except FileExistsError:
            print(f"Folder already exists.")

        log_file_path_base = os.path.join(log_file_path, f"log.log")
        log_file_path = DataSaver.create_log_file_name(base_file_name=log_file_path_base)
        logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s - %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


    @staticmethod
    def physical_monitoring(calibrator, parameters: np.ndarray):
        differences = calibrator.moments_residuals(parameters=parameters)

        # Print latest physical history values
        latest_history = calibrator.physical_history[-1]
        DataSaver.log("\n".join(f" {key}: {value:.5f}" for key, value in latest_history.items()))

        sample_moments_output = DataSaver.format_moments(calibrator.sample_cumulants)
        theoretical_moments_output = DataSaver.format_moments(calibrator.model.theoretical_cumulants())
        relative_errors_output = DataSaver.format_errors(differences, calibrator.sample_cumulants, 'Relative')
        absolute_errors_output = DataSaver.format_errors(differences, calibrator.sample_cumulants, 'Absolute')
        DataSaver.log("\n\n")
        DataSaver.log(f"Iteration: {calibrator.iterations_number}", console=calibrator.console)
        DataSaver.log(f"p_star = {calibrator.p_star}", console=calibrator.console)

        DataSaver.log(
            f"Physical Parameters: {calibrator.physical_history[-1]} \n"
            f"Sample moments: {sample_moments_output}"
            f"\nTheor. moments: {theoretical_moments_output}"
            f"\nRelative errors: {relative_errors_output}"
            f"\nAbsolute errors: {absolute_errors_output}\n", console=calibrator.console)
