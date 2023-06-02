from subprocess import call, run, check_call, check_output
import csv
import pandas as pd
from pathlib import Path
import csv_output_writer

from datetime import datetime
from os.path import exists

run = 1
headers = ['RunNo', 'ExperimentNo', 'Scenario', 'Type', 'InitialTrainingSeed', 'TestingSeed', 'LearningRate', 'DiscountRate', 'Epsilon', 'RewardType', 'MeanReward', 'MeanStopped', 'MeanWaitingTime', 'MeanAverageSpeed', 'MeanAverageSpeed2', 'MeanAverageSpeedCrossCheck']

for index in range(1, 6):
    csv_input_file_path = '../outputs/DE-Enhanced-S3/DE-Enhanced-S3-{}.csv'.format(index)
    csv_output_file_path = '../outputs/DE-Enhanced-S3/DE-Enhanced-S3-{}-reduced.csv'.format(index)
    output_file_content = []
    with open(csv_input_file_path, newline='\n') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            value = row
            print("Before:", end=': ')
            print(value)
            for idx, item in enumerate(value):
                new_item = item
                if "+/-" in new_item:
                    new_item = item[:item.index("+/-")].strip()
                value[idx] = new_item.strip()

            print("End:", end=': ')
            print(value)
            print("----")
            output_file_content.append(value)
    with open(csv_output_file_path, 'w') as file:
        for value in output_file_content:
            file.write(', '.join(value))
            file.write('\n')
        file.close()
