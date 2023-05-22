from subprocess import call, run, check_call, check_output
import csv
from pathlib import Path

def write_headers(csv_file_path, headers):
    Path(Path(csv_file_path).parent).mkdir(parents=True, exist_ok=True)
    with open(csv_file_path, 'a') as file:
        for header in headers:
            file.write(str(header) + ', ')
        file.write('\n')
        file.close()


def write_hyperparameters(csv_file_path, hyperparameters):
    with open(csv_file_path, 'a') as file:
        for h in hyperparameters:
            file.write(str(h) + ', ')
        file.close()


def write_final_values(csv_file_path, metrics):
        with open(csv_file_path, 'a') as file:
            for m in metrics:
                file.write(str(m) + ', ')
            file.write('\n')
            file.close()
