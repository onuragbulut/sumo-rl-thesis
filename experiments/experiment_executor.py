from subprocess import call, run, check_call, check_output
import csv
from pathlib import Path
import csv_output_writer

from datetime import datetime
from os.path import exists

#EXPERIMENT_NO = "E12X"
training_seed_base = 70
testing_seed_base = 80
learning_rates = [0.5] #[0.1, 0.3, 0.5]
discount_rates = [0.99] #[0.8, 0.9, 0.99]
epsilon = [0.2, 0.3, 0.4]
reward_types = ['diff-waiting-time', 'average-speed-2']
#for run in range(1, 6):
run = 5
index = 49
headers = ['RunNo', 'ExperimentNo', 'InitialTrainingSeed', 'TestingSeed', 'LearningRate', 'DiscountRate', 'Epsilon', 'RewardType', 'MeanReward', 'MeanStopped', 'MeanWaitingTime', 'MeanAverageSpeed', 'MeanAverageSpeed2', 'MeanAverageSpeedCrossCheck']
csv_file_path = '../outputs/DoE/DoE-{}.csv'.format(run)
'''
Path(Path(csv_file_path).parent).mkdir(parents=True, exist_ok=True)
with open(csv_file_path, 'w') as file:
    for header in headers:
        file.write(str(header) + ', ')
    file.write('\n')
'''
if not exists(csv_file_path):
    csv_output_writer.write_headers(csv_file_path, headers)

for lr in learning_rates:
    for dr in discount_rates:
        for e in epsilon:
            for rt in reward_types:
                begin = datetime.now()
                EXPERIMENT_NO = "DoE-{}".format(index)
                training_seed = (run * 100) + training_seed_base
                testing_seed = (run * 100) + testing_seed_base
                parameters = ["-runno", run, "-runs", 5, "-experimentno", EXPERIMENT_NO, '-a', lr, '-g', dr, '-e', e, '-r', rt]
                str_args = [str(x) for x in parameters]
                metrics = [run, EXPERIMENT_NO, training_seed, testing_seed, lr, dr, e, rt]
                '''
                for m in metrics:
                    file.write(str(m)+', ')
                file.write('n')
                file.close()
                '''
                #csv_output_writer.write_hyperparameters(csv_file_path, metrics)
                call(["python", "ql_2way-single-intersection.py", '-seed', str(training_seed)] + str_args)
                call(["python", "ql_2way-single-intersection_testing.py", '-seed', str(testing_seed), "-csvfile", csv_file_path] + str_args)
                #call(["python", "ql_2way-single-intersection_baseline.py", '-seed', str(testing_seed)] + str_args)
                index += 1
                delta = (datetime.now() - begin).total_seconds()
                print(f"Experiment took: {int(delta//60)}m {(delta%60):.2f}s")
