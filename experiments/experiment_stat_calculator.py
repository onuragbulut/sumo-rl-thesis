from subprocess import call, run, check_call, check_output
import csv
import pandas as pd
from pathlib import Path
import csv_output_writer

from datetime import datetime
from os.path import exists

run = 1
headers = ['RunNo', 'ExperimentNo', 'Scenario', 'Type', 'InitialTrainingSeed', 'TestingSeed', 'LearningRate', 'DiscountRate', 'Epsilon', 'RewardType', 'MeanReward', 'MeanStopped', 'MeanWaitingTime', 'MeanAverageSpeed', 'MeanAverageSpeed2', 'MeanAverageSpeedCrossCheck']


for scenario in range(1, 9):
    csv_output_file_path = '../outputs/DE-Enhanced/DE-Enhanced-Scenario-{}.csv'.format(scenario)
    output_file_content_test = []
    output_file_content_baseline = []
    output_file_header = []

    for index in range(1, 6):
        csv_input_file_path = '../outputs/DE-Enhanced/DE-Enhanced-{}-reduced.csv'.format(index)

        with open(csv_input_file_path, newline='\n') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for idx_row, row in enumerate(csv_reader):
                if idx_row == 0 and index == 1:
                    value = []
                    for idx, item in enumerate(row):
                        if idx not in [1, 2, 4, 5, 6, 7, 8, 9, 10]:
                            value.append(item)
                            print(item, end=', ')
                            #new_item = item
                            #value[idx] = new_item.strip()
                    output_file_header.append(value)

                elif (idx_row >= ((scenario * 2) - 1) and idx_row <= (scenario * 2)):
                    value = []
                    for idx, item in enumerate(row):
                        if idx not in [1, 2, 4, 5, 6, 7, 8, 9, 10]:
                            value.append(item)
                            print(item, end=', ')
                            #new_item = item
                            #value[idx] = new_item.strip()

                    if idx_row % 2 == 1:
                        output_file_content_test.append(value)
                    else:
                        output_file_content_baseline.append(value)

                    #print("End:", end=': ')
                    #print(value)
                    #print("----")
                    print()

    metrics = ['min', 'max', 'average']
    with open(csv_output_file_path, 'w') as file:
        for value in output_file_header:
            file.write(', '.join(value))
            file.write('\n')

        for value in output_file_content_test:
            file.write(', '.join(value))
            file.write('\n')

        for value in metrics:
            file.write(',')
            file.write(value)
            file.write('\n')

        file.write('\n')

        for value in output_file_content_baseline:
            file.write(', '.join(value))
            file.write('\n')

        for value in metrics:
            file.write(',')
            file.write(value)
            file.write('\n')

        file.close()
