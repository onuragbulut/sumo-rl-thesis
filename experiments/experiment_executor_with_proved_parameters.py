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
run = 5
index = 1
headers = ['RunNo', 'ExperimentNo', 'Scenario', 'Type', 'InitialTrainingSeed', 'TestingSeed', 'LearningRate', 'DiscountRate', 'Epsilon', 'RewardType', 'MeanReward', 'MeanStopped', 'MeanWaitingTime', 'MeanAverageSpeed', 'MeanAverageSpeed2', 'MeanAverageSpeedCrossCheck', 'MeanAverageCO2EmissionTL', 'MeanAverageCOEmissionTL', 'MeanAverageFuelConsumptionTL', 'MeanAverageNoiseEmissionTL', 'MeanAverageCO2Emission', 'MeanAverageCOEmission', 'MeanAverageFuelConsumption', 'MeanAverageNoiseEmission']
csv_file_path = '../outputs/DE-Enhanced-S2/DE-Enhanced-S2-{}.csv'.format(run)
q_table_file_path = '../outputs/2way-single-roundabout_{}/q_table/{}_q_table.txt'.format(EXPERIMENT_NO, EXPERIMENT_NO)
'''
Path(Path(csv_file_path).parent).mkdir(parents=True, exist_ok=True)
with open(csv_file_path, 'w') as file:
    for header in headers:
        file.write(str(header) + ', ')
    file.write('\n')
'''
if not exists(csv_file_path):
    csv_output_writer.write_headers(csv_file_path, headers)


for idx in range(index, 9):
    begin = datetime.now()
    route_file = "../nets/roundabout/input_routes_roundabout_auto_v1-{}.rou.xml".format(idx)
    EXPERIMENT_NO = "DE-Enhanced-S2-{}".format(idx)
    training_seed = (run * 100) + training_seed_base
    testing_seed = (run * 100) + testing_seed_base
    baseline_seed = (run * 100) + testing_seed_base
    parameters = ["-runno", run, "-runs", 5, "-experimentno", EXPERIMENT_NO, '-scenariono', EXPERIMENT_NO, '-a', learning_rate, '-g', discount_rate, '-e', epsilon, '-r', reward_type, '-route', route_file]
    str_args = [str(x) for x in parameters]
    metrics = [run, EXPERIMENT_NO, EXPERIMENT_NO, EXPERIMENT_NO, training_seed, testing_seed, learning_rate, discount_rate, epsilon, reward_type]
    '''
    for m in metrics:
        file.write(str(m)+', ')
    file.write('n')
    file.close()
    '''
    #csv_output_writer.write_hyperparameters(csv_file_path, metrics)
    call(["python", "ql_2way-single-intersection.py", '-seed', str(training_seed)] + str_args)
    call(["python", "ql_2way-single-intersection_testing.py", '-seed', str(testing_seed), "-csvfile", csv_file_path] + str_args)
    call(["python", "ql_2way-single-intersection_baseline.py", '-seed', str(baseline_seed), "-csvfile", csv_file_path] + str_args)
    #index += 1
    delta = (datetime.now() - begin).total_seconds()
    print(f"Experiment took: {int(delta//60)}m {(delta%60):.2f}s")
    print("****")
    print()
