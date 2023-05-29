from subprocess import call, run, check_call, check_output
import csv
from pathlib import Path
import csv_output_writer

from datetime import datetime
from os.path import exists

#EXPERIMENT_NO = "E12X"
training_seed_base = 70
testing_seed_base = 80
baseline_seed_base = 80
learning_rate = 0.5
discount_rate = 0.99
epsilon = 0.2
reward_type = 'diff-waiting-time'
#for run in range(1, 6):
run = 3
index = 1
headers = ['RunNo', 'ExperimentNo', 'Scenario', 'Type', 'InitialTrainingSeed', 'TestingSeed', 'LearningRate', 'DiscountRate', 'Epsilon', 'RewardType', 'MeanReward', 'MeanStopped', 'MeanWaitingTime', 'MeanAverageSpeed', 'MeanAverageSpeed2', 'MeanAverageSpeedCrossCheck', 'MeanAverageCO2EmissionTL', 'MeanAverageCOEmissionTL', 'MeanAverageFuelConsumptionTL', 'MeanAverageNoiseEmissionTL', 'MeanAverageCO2Emission', 'MeanAverageCOEmission', 'MeanAverageFuelConsumption', 'MeanAverageNoiseEmission']
csv_file_path = '../outputs/DE-Enhanced-S3/DE-Enhanced-S3-{}.csv'.format(run)

'''
Path(Path(csv_file_path).parent).mkdir(parents=True, exist_ok=True)
with open(csv_file_path, 'w') as file:
    for header in headers:
        file.write(str(header) + ', ')
    file.write('\n')
'''
if not exists(csv_file_path):
    csv_output_writer.write_headers(csv_file_path, headers)

#q_table_file_path = '../outputs/DE-Enhanced-S3/q_table/q_table_run_{}_scenario_{}.txt'.format(run, idx)
q_table_file_path = '../outputs/DE-Enhanced-S3/q_table/q_table.txt'

for idx in range(index, 9):
    begin = datetime.now()
    route_file = "../nets/roundabout/input_routes_roundabout_auto_v1-{}.rou.xml".format(idx)
    EXPERIMENT_NO = "DE-Enhanced-S3-{}".format(idx)
    training_seed = (run * 100) + training_seed_base
    testing_seed = (run * 100) + testing_seed_base
    baseline_seed = (run * 100) + testing_seed_base
    parameters = ["-runno", run, "-runs", 5, "-experimentno", EXPERIMENT_NO, '-scenariono', EXPERIMENT_NO, '-a', learning_rate, '-g', discount_rate, '-e', epsilon, '-r', reward_type, '-route', route_file, "-qtable", q_table_file_path]
    str_args = [str(x) for x in parameters]
    metrics = [run, EXPERIMENT_NO, EXPERIMENT_NO, EXPERIMENT_NO, training_seed, testing_seed, learning_rate, discount_rate, epsilon, reward_type]
    '''
    for m in metrics:
        file.write(str(m)+', ')
    file.write('n')
    file.close()
    '''
    #csv_output_writer.write_hyperparameters(csv_file_path, metrics)
    if idx == 1 and run == 1:
        call(["python", "ql_2way-single-intersection.py", '-seed', str(training_seed)] + str_args)
    call(["python", "ql_2way-single-intersection_testing.py", '-seed', str(testing_seed), "-csvfile", csv_file_path] + str_args)
    call(["python", "ql_2way-single-intersection_baseline.py", '-seed', str(baseline_seed), "-csvfile", csv_file_path] + str_args)
    #index += 1
    delta = (datetime.now() - begin).total_seconds()
    print(f"Experiment took: {int(delta//60)}m {(delta%60):.2f}s")
    print("****")
    print()
