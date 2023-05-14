import argparse
import os
import sys
from datetime import datetime
import json

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
import numpy as np
from pathlib import Path
import logging
import csv_output_writer

#EXPERIMENT_NO = 'E125'


import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from itertools import cycle

from matplotlib import rc
rc('text', usetex=False)


sns.set(style='darkgrid', rc={'figure.figsize': (10.2, 8.9),
                            'text.usetex': False,
                            'xtick.labelsize': 16,
                            'ytick.labelsize': 16,
                            'font.size': 15,
                            'figure.autolayout': True,
                            'axes.titlesize' : 16,
                            'axes.labelsize' : 15,
                            'lines.linewidth' : 2,
                            'lines.markersize' : 6,
                            'legend.fontsize': 15})
#colors = sns.color_palette("colorblind", 4)
colors = sns.color_palette("bright", 8)
#colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
dashes_styles = cycle(['-', '-.', '--', ':'])
sns.set_palette(colors)
colors = cycle(colors)


def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def draw2(step_list, item_list_train, chart_name, y_label, y_lim=-1, enable_vertical_reference_lines=False, draw_avg=False):
    print("Drawing: {}".format(chart_name))
    #plt.plot(item_list, label=chart_name)

    if draw_avg is True:
        avg_item_list = pd.Series(item_list_train).expanding().mean()
        mean = moving_average(item_list_train, window_size=5)

    fig = plt.figure()
    plt.subplot(2, 1, 1)  # (rows, columns, panel number)
    if len(step_list) > 0:
        plt.plot(step_list, item_list_train, label="train", color='#ff0000', alpha=0.3)
        if draw_avg is True:
            plt.plot(mean, label="train", color='#cc0000', linestyle='--')
    else:
        plt.plot(item_list_train, label="train", color='#ff0000', alpha=0.3)
        if draw_avg is True:
            plt.plot(mean, label="train", color='#cc0000', linestyle='--')

    plt.ylabel(y_label)
    plt.xlabel("Time Step(s)")
    base_max = 0
    if enable_vertical_reference_lines:
        y_max = max(item_list_train)
        plt.yticks(np.arange(0, y_max, 20))
    plt.legend()
    plt.title(chart_name)

    plt.xticks(np.arange(len(item_list_train)), np.arange(1, len(item_list_train)+1, 1))

    plt.legend()
    name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = chart_name.replace(' ', '_').replace(':', '-') + '_' + name + '.pdf'

    final_output = "../outputs/2way-single-roundabout_{}/charts/baseline/{}".format(EXPERIMENT_NO, file_name)
    Path(Path(final_output).parent).mkdir(parents=True, exist_ok=True)
    plt.savefig(final_output, bbox_inches="tight")
    plt.show()


def print_q_table(q_table_param):
    for inner_agent_id in q_table_param.keys():
        current_q_table = q_table_param[inner_agent_id]
        new_q_table = {}
        for key in current_q_table.keys():
            new_q_table[str(key)] = current_q_table[key]

        with open('../outputs/2way-single-roundabout_{}/q_table_{}_agent{}_alpha{}_gamma{}_eps{}_decay{}_reward{}_run{}.txt'.format(EXPERIMENT_NO, experiment_time, agent_id, args.alpha, args.gamma, args.epsilon, args.decay, args.reward, run), 'w') as convert_file:
            convert_file.write(json.dumps(new_q_table))


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    #prs.add_argument("-route", dest="route", type=str, default='../nets/2way-single-intersection/single-intersection-vhvh.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-route", dest="route", type=str, default='../nets/roundabout/input_routes_roundabout_auto_v1.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.4, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=50000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait', required=False, help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("-runno", dest="run_no", type=int, required=True, help="Number of run.\n")
    prs.add_argument("-experimentno", dest="experiment_no", type=str, required=True, help="Number of experiment.\n")
    prs.add_argument("-scenariono", dest="scenario_no", type=str, required=True, help="Number of scenario.\n")
    prs.add_argument("-seed", dest="seed", type=int, required=True, help="Seed.\n")
    prs.add_argument("-csvfile", dest="csv_file", type=str, required=True, help="CSV file path.\n")
    args = prs.parse_args()
    EXPERIMENT_NO = args.experiment_no
    SCENARIO_NO = args.scenario_no
    # metrics = [run, EXPERIMENT_NO, training_seed, testing_seed, lr, dr, e, rt]

    metrics = [args.run_no, EXPERIMENT_NO, args.route, "baseline", "-", args.seed, args.alpha, args.gamma, args.epsilon, args.reward]
    # experiment_time = str(datetime.now()).split('.')[0]
    experiment_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #out_csv = '../outputs/2way-single-intersection/{}_alpha{}_gamma{}_eps{}_decay{}_reward{}'.format(experiment_time, args.alpha, args.gamma, args.epsilon, args.decay, args.reward)
    out_csv = '../outputs/2way-single-roundabout_{}/baseline/{}_baseline_{}_alpha{}_gamma{}_eps{}_decay{}_reward{}'.format(EXPERIMENT_NO, EXPERIMENT_NO, experiment_time,
                                                                                                     args.alpha,
                                                                                                     args.gamma,
                                                                                                     args.epsilon,
                                                                                                     args.decay,
                                                                                                     args.reward)

    #env = SumoEnvironment(net_file='../nets/2way-single-intersection/single-intersection.net.xml',
    env = SumoEnvironment(net_file='../nets/roundabout/net_roundabout_WORKING_single_light_2.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          use_gui=args.gui,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          reward_fn='queue',
                          sumo_warnings=False)

    final_log_file = "../outputs/2way-single-roundabout_{}/log/{}-metrics-baseline.log".format(EXPERIMENT_NO,
                                                                                             EXPERIMENT_NO)
    Path(Path(final_log_file).parent).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(handlers=[logging.FileHandler(filename=final_log_file, encoding='utf-8', mode='a+')],
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%F %A %T",
                        level=logging.INFO)

    q_table = {}
    episode_rewards = []
    J0_stopped_list = []
    J0_stopped_cross_check_list = []
    J0_acc_wt_list = []
    J0_avg_speed_list = []
    J0_avg_speed_2_list = []
    J0_avg_speed_cross_check_list = []

    ####
    J0_average_co2_emission_tl_list = []
    J0_average_co_emission_tl_list = []
    J0_average_fuel_consumption_tl_list = []
    J0_average_noise_emission_tl_list = []

    J0_average_co2_emission_list = []
    J0_average_co_emission_list = []
    J0_average_fuel_consumptions_list = []
    J0_average_noise_emissions_list = []
    ####

    ts = 'J0'
    missing_state_count = 0
    total_count = 0

    inner_outer_total_rewards = []
    all_per_episode_rewards = []

    inner_outer_total_stopped = []
    all_per_episode_stopped = []

    inner_outer_total_stopped_cross_check = []
    all_per_episode_stopped_cross_check = []

    inner_outer_total_wt = []
    all_per_episode_wt = []

    inner_outer_total_avg_speed = []
    all_per_episode_avg_speed = []

    inner_outer_total_avg_speed_2 = []
    all_per_episode_avg_speed_2 = []

    inner_outer_total_avg_speed_cross_check = []
    all_per_episode_avg_speed_cross_check = []

    ####
    inner_outer_total_average_co2_emission_tl = []
    all_per_episode_average_co2_emission_tl = []

    inner_outer_total_average_co_emission_tl = []
    all_per_episode_average_co_emission_tl = []

    inner_outer_total_average_fuel_consumption_tl = []
    all_per_episode_average_fuel_consumption_tl = []

    inner_outer_total_average_noise_emission_tl = []
    all_per_episode_average_noise_emission_tl = []

    inner_outer_total_average_co2_emission = []
    all_per_episode_average_co2_emission = []

    inner_outer_total_average_co_emission = []
    all_per_episode_average_co_emission = []

    inner_outer_total_average_fuel_consumptions = []
    all_per_episode_average_fuel_consumptions = []

    inner_outer_total_average_noise_emissions = []
    all_per_episode_average_noise_emissions = []
    ####

    #seed = 80
    seed = args.seed
    print("Run: {} for Baseline experiment: {} - route: {} - first seed: {} - alpha: {} - gamma: {} - epsilon: {} - reward: {}".format(args.run_no, EXPERIMENT_NO, args.route, seed, args.alpha, args.gamma, args.epsilon, args.reward))
    logging.info("Run: {} for Baseline experiment: {} - route: {} - first seed: {} - alpha: {} - gamma: {} - epsilon: {} - reward: {}".format(args.run_no, EXPERIMENT_NO, args.route, seed, args.alpha, args.gamma, args.epsilon, args.reward))
    #for run in range(1, args.runs+1):
    for run in range(1, 2):
        initial_states = env.reset(seed=seed)
        state_ = env.encode(initial_states[ts], ts)
        step = 0
        done = {'__all__': False}
        total_rewards_ep = 0
        J0_stopped_ep = 0
        J0_stopped_cross_check_ep = 0
        J0_acc_wt_ep = 0
        J0_avg_speed_ep = 0
        J0_avg_speed_2_ep = 0
        J0_avg_speed_cross_check_ep = 0

        ####
        J0_average_co2_emission_tl_ep = 0
        J0_average_co_emission_tl_ep = 0
        J0_average_fuel_consumption_tl_ep = 0
        J0_average_noise_emission_tl_ep = 0

        J0_average_co2_emission_ep = 0
        J0_average_co_emission_ep = 0
        J0_average_fuel_consumptions_ep = 0
        J0_average_noise_emissions_ep = 0
        ####

        infos = []
        action = 0
        per_episode_rewards = []
        per_episode_stopped = []
        per_episode_stopped_cross_check = []
        per_episode_avg_speed = []
        per_episode_avg_speed_2 = []
        per_episode_avg_speed_cross_check = []

        ####
        per_episode_average_co2_emission_tl = []
        per_episode_average_co_emission_tl = []
        per_episode_average_fuel_consumption_tl = []
        per_episode_average_noise_emission_tl = []

        per_episode_average_co2_emission = []
        per_episode_average_co_emission = []
        per_episode_average_fuel_consumptions = []
        per_episode_average_noise_emissions = []
        ####

        per_episode_wt = []
        while not done['__all__']:
            actions = {ts: action}
            s, r, done, infos = env.step(action=actions)
            #img = env.render()
            total_count += 1
            total_rewards_ep += r[ts]
            J0_stopped_ep += infos['J0_stopped']
            J0_stopped_cross_check_ep += infos['J0_stopped_cross_check']
            J0_acc_wt_ep += infos['J0_accumulated_waiting_time']
            J0_avg_speed_ep += infos['J0_average_speed']
            J0_avg_speed_2_ep += infos['J0_average_speed_2']
            J0_avg_speed_cross_check_ep += infos['J0_average_speed_cross_check']

            ####
            J0_average_co2_emission_tl_ep += infos['J0_average_co2_emission_tl']
            J0_average_co_emission_tl_ep += infos['J0_average_co_emission_tl']
            J0_average_fuel_consumption_tl_ep += infos['J0_average_fuel_consumption_tl']
            J0_average_noise_emission_tl_ep += infos['J0_average_noise_emission_tl']

            J0_average_co2_emission_ep += infos['J0_average_co2_emission']
            J0_average_co_emission_ep += infos['J0_average_co_emission']
            J0_average_fuel_consumptions_ep += infos['J0_average_fuel_consumptions']
            J0_average_noise_emissions_ep += infos['J0_average_noise_emissions']
            ####

            action += 1
            action %= 4

            inner_outer_total_rewards.append(r[ts])
            per_episode_rewards.append(r[ts])

            inner_outer_total_stopped.append(infos['J0_stopped'])
            per_episode_stopped.append(infos['J0_stopped'])

            inner_outer_total_stopped_cross_check.append(infos['J0_stopped_cross_check'])
            per_episode_stopped_cross_check.append(infos['J0_stopped_cross_check'])

            inner_outer_total_wt.append(infos['J0_accumulated_waiting_time'])
            per_episode_wt.append(infos['J0_accumulated_waiting_time'])

            inner_outer_total_avg_speed.append(infos['J0_average_speed'])
            per_episode_avg_speed.append(infos['J0_average_speed'])

            inner_outer_total_avg_speed_2.append(infos['J0_average_speed_2'])
            per_episode_avg_speed_2.append(infos['J0_average_speed_2'])

            inner_outer_total_avg_speed_cross_check.append(infos['J0_average_speed_cross_check'])
            per_episode_avg_speed_cross_check.append(infos['J0_average_speed_cross_check'])

            ####
            inner_outer_total_average_co2_emission_tl.append(infos['J0_average_co2_emission_tl'])
            per_episode_average_co2_emission_tl.append(infos['J0_average_co2_emission_tl'])

            inner_outer_total_average_co_emission_tl.append(infos['J0_average_co_emission_tl'])
            per_episode_average_co_emission_tl.append(infos['J0_average_co_emission_tl'])

            inner_outer_total_average_fuel_consumption_tl.append(infos['J0_average_fuel_consumption_tl'])
            per_episode_average_fuel_consumption_tl.append(infos['J0_average_fuel_consumption_tl'])

            inner_outer_total_average_noise_emission_tl.append(infos['J0_average_noise_emission_tl'])
            per_episode_average_noise_emission_tl.append(infos['J0_average_noise_emission_tl'])

            inner_outer_total_average_co2_emission.append(infos['J0_average_co2_emission'])
            per_episode_average_co2_emission.append(infos['J0_average_co2_emission'])

            inner_outer_total_average_co_emission.append(infos['J0_average_co_emission'])
            per_episode_average_co_emission.append(infos['J0_average_co_emission'])

            inner_outer_total_average_fuel_consumptions.append(infos['J0_average_fuel_consumptions'])
            per_episode_average_fuel_consumptions.append(infos['J0_average_fuel_consumptions'])

            inner_outer_total_average_noise_emissions.append(infos['J0_average_noise_emissions'])
            per_episode_average_noise_emissions.append(infos['J0_average_noise_emissions'])
            ####

            state_ = env.encode(s[ts], ts)
        episode_rewards.append(total_rewards_ep)
        J0_stopped_list.append(J0_stopped_ep)
        J0_stopped_cross_check_list.append(J0_stopped_cross_check_ep)
        J0_acc_wt_list.append(J0_acc_wt_ep)
        J0_avg_speed_list.append(J0_avg_speed_ep)
        J0_avg_speed_2_list.append(J0_avg_speed_2_ep)
        J0_avg_speed_cross_check_list.append(J0_avg_speed_cross_check_ep)

        ####
        J0_average_co2_emission_tl_list.append(J0_average_co2_emission_tl_ep)
        J0_average_co_emission_tl_list.append(J0_average_co_emission_tl_ep)
        J0_average_fuel_consumption_tl_list.append(J0_average_fuel_consumption_tl_ep)
        J0_average_noise_emission_tl_list.append(J0_average_noise_emission_tl_ep)

        J0_average_co2_emission_list.append(J0_average_co2_emission_ep)
        J0_average_co_emission_list.append(J0_average_co_emission_ep)
        J0_average_fuel_consumptions_list.append(J0_average_fuel_consumptions_ep)
        J0_average_noise_emissions_list.append(J0_average_noise_emissions_ep)
        ####

        all_per_episode_rewards.append(per_episode_rewards)
        all_per_episode_stopped.append(per_episode_stopped)
        all_per_episode_stopped_cross_check.append(per_episode_stopped_cross_check)
        all_per_episode_wt.append(per_episode_wt)
        all_per_episode_avg_speed.append(per_episode_avg_speed)
        all_per_episode_avg_speed_2.append(per_episode_avg_speed_2)
        all_per_episode_avg_speed_cross_check.append(per_episode_avg_speed_cross_check)

        ####
        all_per_episode_average_co2_emission_tl.append(per_episode_average_co2_emission_tl)
        all_per_episode_average_co_emission_tl.append(per_episode_average_co_emission_tl)
        all_per_episode_average_fuel_consumption_tl.append(per_episode_average_fuel_consumption_tl)
        all_per_episode_average_noise_emission_tl.append(per_episode_average_noise_emission_tl)
        all_per_episode_average_co2_emission.append(per_episode_average_co2_emission)
        all_per_episode_average_co_emission.append(per_episode_average_co_emission)
        all_per_episode_average_fuel_consumptions.append(per_episode_average_fuel_consumptions)
        all_per_episode_average_noise_emissions.append(per_episode_average_noise_emissions)
        ####

        env.save_csv(out_csv, run)
        env.close()

    logging.info("Total Missing States: {}".format(missing_state_count))

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    #print(f"Mean Reward={mean_reward:.2f} +/- {std_reward:.2f}")
    #print(f"Sum Reward={np.sum(episode_rewards):.2f}, {total_count:.2f} {(np.sum(episode_rewards)/total_count):.2f}")
    logging.info(f"Mean Reward={mean_reward:.2f} +/- {std_reward:.2f}")

    mean_stopped = np.mean(J0_stopped_list)
    std_stopped = np.std(J0_stopped_list)
    #print(f"Mean Stopped={mean_stopped:.2f} +/- {std_stopped:.2f}")
    #print(f"Sum Stopped={np.sum(J0_stopped_list):.2f}, {total_count:.2f} {(np.sum(J0_stopped_list) / total_count):.2f}")
    logging.info(f"Mean Stopped={mean_stopped:.2f} +/- {std_stopped:.2f}")

    mean_stopped_cross_check = np.mean(J0_stopped_cross_check_list)
    std_stopped_cross_check = np.std(J0_stopped_cross_check_list)
    #print(f"Mean_Stopped_Cross_Check={mean_stopped_cross_check:.2f} +/- {std_stopped_cross_check:.2f}")
    logging.info(f"Mean_Stopped={mean_stopped_cross_check:.2f} +/- {std_stopped_cross_check:.2f}")

    mean_acc_wt = np.mean(J0_acc_wt_list)
    std_acc_wt = np.std(J0_acc_wt_list)
    #print(f"Mean Waiting Time={mean_acc_wt:.2f} +/- {std_acc_wt:.2f}")
    #print(f"Sum Waiting Time={np.sum(J0_acc_wt_list):.2f}, {total_count:.2f} {(np.sum(J0_acc_wt_list) / total_count):.2f}")
    logging.info(f"Mean Waiting Time={mean_acc_wt:.2f} +/- {std_acc_wt:.2f}")

    mean_avg_speed = np.mean(J0_avg_speed_list)
    std_avg_speed = np.std(J0_avg_speed_list)
    #print(f"Mean_avg_speed={mean_avg_speed:.2f} +/- {std_avg_speed:.2f}")
    logging.info(f"Mean_avg_speed={mean_avg_speed:.2f} +/- {std_avg_speed:.2f}")

    mean_avg_speed_2 = np.mean(J0_avg_speed_2_list)
    std_avg_speed_2 = np.std(J0_avg_speed_2_list)
    #print(f"Mean_avg_speed_2={mean_avg_speed_2:.2f} +/- {std_avg_speed_2:.2f}")
    logging.info(f"Mean_avg_speed_2={mean_avg_speed_2:.2f} +/- {std_avg_speed_2:.2f}")

    mean_avg_speed_cross_check = np.mean(J0_avg_speed_cross_check_list)
    std_avg_speed_cross_check = np.std(J0_avg_speed_cross_check_list)
    #print(f"Mean_avg_speed_cross_check={mean_avg_speed_cross_check:.2f} +/- {std_avg_speed_cross_check:.2f}")
    logging.info(f"Mean_avg_speed_cross_check={mean_avg_speed_cross_check:.2f} +/- {std_avg_speed_cross_check:.2f}")

    ####
    mean_average_co2_emission_tl = np.mean(J0_average_co2_emission_tl_list)
    std_average_co2_emission_tl = np.std(J0_average_co2_emission_tl_list)
    # print(f"Mean_average_co2_emission_tl={mean_average_co2_emission_tl:.2f} +/- {std_average_co2_emission_tl:.2f}")
    logging.info(f"Mean_average_co2_emission_tl={mean_average_co2_emission_tl:.2f} +/- {std_average_co2_emission_tl:.2f}")

    mean_average_co_emission_tl = np.mean(J0_average_co_emission_tl_list)
    std_average_co_emission_tl = np.std(J0_average_co_emission_tl_list)
    # print(f"Mean_average_co_emission_tl={mean_average_co_emission_tl:.2f} +/- {std_average_co_emission_tl:.2f}")
    logging.info(f"Mean_average_co_emission_tl={mean_average_co_emission_tl:.2f} +/- {std_average_co_emission_tl:.2f}")

    mean_average_fuel_consumption_tl = np.mean(J0_average_fuel_consumption_tl_list)
    std_average_fuel_consumption_tl = np.std(J0_average_fuel_consumption_tl_list)
    # print(f"Mean_average_fuel_consumption_tl={mean_average_fuel_consumption_tl:.2f} +/- {std_average_fuel_consumption_tl:.2f}")
    logging.info(f"Mean_average_fuel_consumption_tl={mean_average_fuel_consumption_tl:.2f} +/- {std_average_fuel_consumption_tl:.2f}")

    mean_average_noise_emission_tl = np.mean(J0_average_noise_emission_tl_list)
    std_average_noise_emission_tl = np.std(J0_average_noise_emission_tl_list)
    # print(f"Mean_average_noise_emission_tl={mean_average_noise_emission_tl:.2f} +/- {std_average_noise_emission_tl:.2f}")
    logging.info(f"Mean_average_noise_emission_tl={mean_average_noise_emission_tl:.2f} +/- {std_average_noise_emission_tl:.2f}")

    mean_average_co2_emission = np.mean(J0_average_co2_emission_list)
    std_average_co2_emission = np.std(J0_average_co2_emission_list)
    # print(f"Mean_average_co2_emission={mean_average_co2_emission:.2f} +/- {std_average_co2_emission:.2f}")
    logging.info(f"Mean_average_co2_emission={mean_average_co2_emission:.2f} +/- {std_average_co2_emission:.2f}")

    mean_average_co_emission = np.mean(J0_average_co_emission_list)
    std_average_co_emission = np.std(J0_average_co_emission_list)
    # print(f"Mean_average_co_emission={mean_average_co_emission:.2f} +/- {std_average_co_emission:.2f}")
    logging.info(f"Mean_average_co_emission={mean_average_co_emission:.2f} +/- {std_average_co_emission:.2f}")

    mean_average_fuel_consumptions = np.mean(J0_average_fuel_consumptions_list)
    std_average_fuel_consumptions = np.std(J0_average_fuel_consumptions_list)
    # print(f"Mean_average_fuel_consumption_tl={mean_average_fuel_consumptions:.2f} +/- {std_average_fuel_consumptions:.2f}")
    logging.info(f"Mean_average_fuel_consumptions={mean_average_fuel_consumptions:.2f} +/- {std_average_fuel_consumptions:.2f}")

    mean_average_noise_emissions = np.mean(J0_average_noise_emissions_list)
    std_average_noise_emissions = np.std(J0_average_noise_emissions_list)
    # print(f"Mean_average_noise_emissions={mean_average_noise_emissions:.2f} +/- {std_average_noise_emissions:.2f}")
    logging.info(f"Mean_average_noise_emissions={mean_average_noise_emissions:.2f} +/- {std_average_noise_emissions:.2f}")
    ####

    #print("****************************************")
    logging.info("****************************************")
    #print("Calculation 1 - Overall")
    logging.info("Calculation 1 - Overall ")
    #print("****************************************")
    logging.info("****************************************")
    inner_outer_mean_reward = np.mean(inner_outer_total_rewards)
    inner_outer_std_reward = np.std(inner_outer_total_rewards)
    #print(f"Inner X Outer Loops Mean Reward={inner_outer_mean_reward:.3f} +/- {inner_outer_std_reward:.3f}")
    logging.info(f"Inner X Outer Loops Mean Reward={inner_outer_mean_reward:.3f} +/- {inner_outer_std_reward:.3f}")
    metrics.append(f"{inner_outer_mean_reward:.3f} +/- {inner_outer_std_reward:.3f}")
    # Stopped
    inner_outer_mean_stopped = np.mean(inner_outer_total_stopped)
    inner_outer_std_stopped = np.std(inner_outer_total_stopped)
    #print(f"Inner X Outer Loops Mean Stopped= {inner_outer_mean_stopped:.3f} +/- {inner_outer_std_stopped:.3f}")
    logging.info(f"Inner X Outer Loops Mean Stopped= {inner_outer_mean_stopped:.3f} +/- {inner_outer_std_stopped:.3f}")
    metrics.append(f"{inner_outer_mean_stopped:.3f} +/- {inner_outer_std_stopped:.3f}")
    # Stopped Cross Check
    inner_outer_mean_stopped_cross_check = np.mean(inner_outer_total_stopped_cross_check)
    inner_outer_std_stopped_cross_check = np.std(inner_outer_total_stopped_cross_check)
    #print(f"Inner X Outer Loops Mean Stopped Cross Check= {inner_outer_mean_stopped_cross_check:.3f} +/- {inner_outer_std_stopped_cross_check:.3f}")
    logging.info(f"Inner X Outer Loops Mean Stopped Cross Check= {inner_outer_mean_stopped_cross_check:.3f} +/- {inner_outer_std_stopped_cross_check:.3f}")

    # Waiting Time
    inner_outer_mean_wt = np.mean(inner_outer_total_wt)
    inner_outer_std_wt = np.std(inner_outer_total_wt)
    #print(f"Inner X Outer Loops Mean Waiting Time= {inner_outer_mean_wt:.3f} +/- {inner_outer_std_wt:.3f}")
    logging.info(f"Inner X Outer Loops Mean Waiting Time= {inner_outer_mean_wt:.3f} +/- {inner_outer_std_wt:.3f}")
    metrics.append(f"{inner_outer_mean_wt:.3f} +/- {inner_outer_std_wt:.3f}")
    # Avg Speed
    inner_outer_mean_avg_speed = np.mean(inner_outer_total_avg_speed)
    inner_outer_std_avg_speed = np.std(inner_outer_total_avg_speed)
    #print(f"Inner X Outer Loops Mean Avg Speed= {inner_outer_mean_avg_speed:.3f} +/- {inner_outer_std_avg_speed:.3f}")
    logging.info(f"Inner X Outer Loops Mean Avg Speed= {inner_outer_mean_avg_speed:.3f} +/- {inner_outer_std_avg_speed:.3f}")
    metrics.append(f"{inner_outer_mean_avg_speed:.3f} +/- {inner_outer_std_avg_speed:.3f}")
    # Avg Speed 2
    inner_outer_mean_avg_speed_2 = np.mean(inner_outer_total_avg_speed_2)
    inner_outer_std_avg_speed_2 = np.std(inner_outer_total_avg_speed_2)
    #print(f"Inner X Outer Loops Mean Avg Speed 2= {inner_outer_mean_avg_speed_2:.3f} +/- {inner_outer_std_avg_speed_2:.3f}")
    logging.info(f"Inner X Outer Loops Mean Avg Speed 2= {inner_outer_mean_avg_speed_2:.3f} +/- {inner_outer_std_avg_speed_2:.3f}")
    metrics.append(f"{inner_outer_mean_avg_speed_2:.3f} +/- {inner_outer_std_avg_speed_2:.3f}")
    # Avg Speed Cross Check
    inner_outer_mean_avg_speed_cross_check = np.mean(inner_outer_total_avg_speed_cross_check)
    inner_outer_std_avg_speed_cross_check = np.std(inner_outer_total_avg_speed_cross_check)
    #print(f"Inner X Outer Loops Mean Avg Speed Cross Check= {inner_outer_mean_avg_speed_cross_check:.3f} +/- {inner_outer_std_avg_speed_cross_check:.3f}")
    logging.info(f"Inner X Outer Loops Mean Avg Speed Cross Check= {inner_outer_mean_avg_speed_cross_check:.3f} +/- {inner_outer_std_avg_speed_cross_check:.3f}")
    metrics.append(f"{inner_outer_mean_avg_speed_cross_check:.3f} +/- {inner_outer_std_avg_speed_cross_check:.3f}")

    ####
    # Avg CO2 Emission TL
    inner_outer_mean_average_co2_emission_tl = np.mean(inner_outer_total_average_co2_emission_tl)
    inner_outer_std_average_co2_emission_tl = np.std(inner_outer_total_average_co2_emission_tl)
    # print(f"Inner X Outer Loops Mean Average CO2 Emission TL= {inner_outer_mean_average_co2_emission_tl:.3f} +/- {inner_outer_std_average_co2_emission_tl:.3f}")
    logging.info(f"Inner X Outer Loops Mean Average CO2 Emission TL= {inner_outer_mean_average_co2_emission_tl:.3f} +/- {inner_outer_std_average_co2_emission_tl:.3f}")
    metrics.append(f"{inner_outer_mean_average_co2_emission_tl:.3f} +/- {inner_outer_std_average_co2_emission_tl:.3f}")

    # Avg CO Emission TL
    inner_outer_mean_average_co_emission_tl = np.mean(inner_outer_total_average_co_emission_tl)
    inner_outer_std_average_co_emission_tl = np.std(inner_outer_total_average_co_emission_tl)
    # print(f"Inner X Outer Loops Mean Average CO Emission TL= {inner_outer_mean_average_co_emission_tl:.3f} +/- {inner_outer_std_average_co_emission_tl:.3f}")
    logging.info(f"Inner X Outer Loops Mean Average CO Emission TL= {inner_outer_mean_average_co_emission_tl:.3f} +/- {inner_outer_std_average_co_emission_tl:.3f}")
    metrics.append(f"{inner_outer_mean_average_co_emission_tl:.3f} +/- {inner_outer_std_average_co_emission_tl:.3f}")

    # Avg Fuel Consumptions TL
    inner_outer_mean_average_fuel_consumption_tl = np.mean(inner_outer_total_average_fuel_consumption_tl)
    inner_outer_std_average_fuel_consumption_tl = np.std(inner_outer_total_average_fuel_consumption_tl)
    # print(f"Inner X Outer Loops Mean Average Fuel Consumption TL= {inner_outer_mean_average_fuel_consumption_tl:.3f} +/- {inner_outer_std_average_fuel_consumption_tl:.3f}")
    logging.info(f"Inner X Outer Loops Mean Average Fuel Consumption TL= {inner_outer_mean_average_fuel_consumption_tl:.3f} +/- {inner_outer_std_average_fuel_consumption_tl:.3f}")
    metrics.append(f"{inner_outer_mean_average_fuel_consumption_tl:.3f} +/- {inner_outer_std_average_fuel_consumption_tl:.3f}")

    # Avg Noise Emission TL
    inner_outer_mean_average_noise_emission_tl = np.mean(inner_outer_total_average_noise_emission_tl)
    inner_outer_std_average_noise_emission_tl = np.std(inner_outer_total_average_noise_emission_tl)
    # print(f"Inner X Outer Loops Mean Average Noise Emission TL= {inner_outer_mean_average_noise_emission_tl:.3f} +/- {inner_outer_std_average_noise_emission_tl:.3f}")
    logging.info(f"Inner X Outer Loops Mean Average Noise Emission TL= {inner_outer_mean_average_noise_emission_tl:.3f} +/- {inner_outer_std_average_noise_emission_tl:.3f}")
    metrics.append(f"{inner_outer_mean_average_noise_emission_tl:.3f} +/- {inner_outer_std_average_noise_emission_tl:.3f}")

    # Avg CO2 Emission
    inner_outer_mean_average_co2_emission = np.mean(inner_outer_total_average_co2_emission)
    inner_outer_std_average_co2_emission = np.std(inner_outer_total_average_co2_emission)
    # print(f"Inner X Outer Loops Mean Average CO2 Emission= {inner_outer_mean_average_co2_emission:.3f} +/- {inner_outer_std_average_co2_emission:.3f}")
    logging.info(f"Inner X Outer Loops Mean Average CO2 Emission= {inner_outer_mean_average_co2_emission:.3f} +/- {inner_outer_std_average_co2_emission:.3f}")
    metrics.append(f"{inner_outer_mean_average_co2_emission:.3f} +/- {inner_outer_std_average_co2_emission:.3f}")

    # Avg CO Emission
    inner_outer_mean_average_co_emission = np.mean(inner_outer_total_average_co_emission)
    inner_outer_std_average_co_emission = np.std(inner_outer_total_average_co_emission)
    # print(f"Inner X Outer Loops Mean Average CO Emission= {inner_outer_mean_average_co_emission:.3f} +/- {inner_outer_std_average_co_emission:.3f}")
    logging.info(f"Inner X Outer Loops Mean Average CO Emission= {inner_outer_mean_average_co_emission:.3f} +/- {inner_outer_std_average_co_emission:.3f}")
    metrics.append(f"{inner_outer_mean_average_co_emission:.3f} +/- {inner_outer_std_average_co_emission:.3f}")

    # Avg Fuel Consumptions
    inner_outer_mean_average_fuel_consumptions = np.mean(inner_outer_total_average_fuel_consumptions)
    inner_outer_std_average_fuel_consumptions = np.std(inner_outer_total_average_fuel_consumptions)
    # print(f"Inner X Outer Loops Mean Average Fuel Consumptions= {inner_outer_mean_average_fuel_consumptions:.3f} +/- {inner_outer_std_average_fuel_consumptions:.3f}")
    logging.info(f"Inner X Outer Loops Mean Average Fuel Consumptions= {inner_outer_mean_average_fuel_consumptions:.3f} +/- {inner_outer_std_average_fuel_consumptions:.3f}")
    metrics.append(f"{inner_outer_mean_average_fuel_consumptions:.3f} +/- {inner_outer_std_average_fuel_consumptions:.3f}")

    # Avg Noise Emission
    inner_outer_mean_average_noise_emissions = np.mean(inner_outer_total_average_noise_emissions)
    inner_outer_std_average_noise_emissions = np.std(inner_outer_total_average_noise_emissions)
    # print(f"Inner X Outer Loops Mean Average Noise Emissions= {inner_outer_mean_average_noise_emissions:.3f} +/- {inner_outer_std_average_noise_emissions:.3f}")
    logging.info(f"Inner X Outer Loops Mean Average Noise Emissions= {inner_outer_mean_average_noise_emissions:.3f} +/- {inner_outer_std_average_noise_emissions:.3f}")
    metrics.append(f"{inner_outer_mean_average_noise_emissions:.3f} +/- {inner_outer_std_average_noise_emissions:.3f}")
    ####

    #print("****************************************")
    logging.info("****************************************")
    #print("Calculation 2 - Per Episode")
    logging.info("Calculation 2 - Per Episode")
    #print("****************************************")
    logging.info("****************************************")

    result_list = []
    std_list = []
    for index, elem in enumerate(all_per_episode_rewards):
        mean = np.mean(elem)
        std = np.std(elem)
        result_list.append(mean)
        std_list.append(std)
        #print(f"Episode:{index} Mean Waiting Time={mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Waiting Time={mean:.3f} +/- {std:.3f}")

    #print(f"1-Total Mean Reward= {np.mean(result_list):.3f} +/- {np.std(result_list):.3f}")
    #print(f"2-Total Mean Reward= {np.mean(result_list):.3f} +/- {np.mean(std_list):.3f}")
    logging.info(f"1-Total Mean Reward= {np.mean(result_list):.3f} +/- {np.std(result_list):.3f}")
    logging.info(f"2-Total Mean Reward= {np.mean(result_list):.3f} +/- {np.mean(std_list):.3f}")
    #print("****")
    logging.info("****")

    # Stopped
    result_stopped_list = []
    std_stopped_list = []
    for index, elem in enumerate(all_per_episode_stopped):
        mean = np.mean(elem)
        std = np.std(elem)
        result_stopped_list.append(mean)
        std_stopped_list.append(std)
        #print(f"Episode:{index} Mean Stopped= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Stopped= {mean:.3f} +/- {std:.3f}")

    #print(f"1-Total Mean Stopped= {np.mean(result_stopped_list):.3f} +/- {np.std(result_stopped_list):.3f}")
    #print(f"2-Total Mean Stopped= {np.mean(result_stopped_list):.3f} +/- {np.mean(std_stopped_list):.3f}")
    logging.info(f"1-Total Mean Stopped= {np.mean(result_stopped_list):.3f} +/- {np.std(result_stopped_list):.3f}")
    logging.info(f"2-Total Mean Stopped= {np.mean(result_stopped_list):.3f} +/- {np.mean(std_stopped_list):.3f}")
    #print("****")
    logging.info("****")

    # Stopped Cross Check
    result_stopped_cross_check_list = []
    std_stopped_cross_check_list = []
    for index, elem in enumerate(all_per_episode_stopped_cross_check):
        mean = np.mean(elem)
        std = np.std(elem)
        result_stopped_cross_check_list.append(mean)
        std_stopped_cross_check_list.append(std)
        #print(f"Episode:{index} Mean Stopped Cross Check= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Stopped Cross Check= {mean:.3f} +/- {std:.3f}")

    #print(f"1-Total Mean Stopped Cross Check= {np.mean(result_stopped_cross_check_list):.3f} +/- {np.std(result_stopped_cross_check_list):.3f}")
    #print(f"2-Total Mean Stopped Cross Check= {np.mean(result_stopped_cross_check_list):.3f} +/- {np.mean(std_stopped_cross_check_list):.3f}")
    logging.info(f"1-Total Mean Stopped Cross Check= {np.mean(result_stopped_cross_check_list):.3f} +/- {np.std(result_stopped_cross_check_list):.3f}")
    logging.info(f"2-Total Mean Stopped Cross Check= {np.mean(result_stopped_cross_check_list):.3f} +/- {np.mean(std_stopped_cross_check_list):.3f}")
    #print("****")
    logging.info("****")

    # Waiting Time
    result_wt_list = []
    std_wt_list = []
    for index, elem in enumerate(all_per_episode_wt):
        mean = np.mean(elem)
        std = np.std(elem)
        result_wt_list.append(mean)
        std_wt_list.append(std)
        #print(f"Episode:{index} Mean Waiting Time= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Waiting Time= {mean:.3f} +/- {std:.3f}")

    #print(f"1-Total Mean Waiting Time= {np.mean(result_wt_list):.3f} +/- {np.std(result_wt_list):.3f}")
    #print(f"2-Total Mean Waiting Time= {np.mean(result_wt_list):.3f} +/- {np.mean(std_wt_list):.3f}")
    logging.info(f"1-Total Mean Waiting Time= {np.mean(result_wt_list):.3f} +/- {np.std(result_wt_list):.3f}")
    logging.info(f"2-Total Mean Waiting Time= {np.mean(result_wt_list):.3f} +/- {np.mean(std_wt_list):.3f}")

    # Avg Speed
    result_avg_speed_list = []
    std_avg_speed_list = []
    for index, elem in enumerate(all_per_episode_avg_speed):
        mean = np.mean(elem)
        std = np.std(elem)
        result_avg_speed_list.append(mean)
        std_avg_speed_list.append(std)
        #print(f"Episode:{index} Mean Avg Speed= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Avg Speed= {mean:.3f} +/- {std:.3f}")

    #print(f"1-Total Mean Avg Speed= {np.mean(result_avg_speed_list):.3f} +/- {np.std(result_avg_speed_list):.3f}")
    #print(f"2-Total Mean Avg Speed= {np.mean(result_avg_speed_list):.3f} +/- {np.mean(std_avg_speed_list):.3f}")
    logging.info(f"1-Total Mean Avg Speed= {np.mean(result_avg_speed_list):.3f} +/- {np.std(result_avg_speed_list):.3f}")
    logging.info(f"2-Total Mean Avg Speed= {np.mean(result_avg_speed_list):.3f} +/- {np.mean(std_avg_speed_list):.3f}")

    # Avg Speed 2
    result_avg_speed_2_list = []
    std_avg_speed_2_list = []
    for index, elem in enumerate(all_per_episode_avg_speed_2):
        mean = np.mean(elem)
        std = np.std(elem)
        result_avg_speed_2_list.append(mean)
        std_avg_speed_2_list.append(std)
        #print(f"Episode:{index} Mean Avg Speed 2= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Avg Speed 2= {mean:.3f} +/- {std:.3f}")

    #print(f"1-Total Mean Avg Speed 2= {np.mean(result_avg_speed_2_list):.3f} +/- {np.std(result_avg_speed_2_list):.3f}")
    #print(f"2-Total Mean Avg Speed 2= {np.mean(result_avg_speed_2_list):.3f} +/- {np.mean(std_avg_speed_2_list):.3f}")
    logging.info(f"1-Total Mean Avg Speed 2= {np.mean(result_avg_speed_2_list):.3f} +/- {np.std(result_avg_speed_2_list):.3f}")
    logging.info(f"2-Total Mean Avg Speed 2= {np.mean(result_avg_speed_2_list):.3f} +/- {np.mean(std_avg_speed_2_list):.3f}")

    # Avg Speed Cross Check
    result_avg_speed_cross_check_list = []
    std_avg_speed_cross_check_list = []
    for index, elem in enumerate(all_per_episode_avg_speed_cross_check):
        mean = np.mean(elem)
        std = np.std(elem)
        result_avg_speed_cross_check_list.append(mean)
        std_avg_speed_cross_check_list.append(std)
        #print(f"Episode:{index} Mean Avg Speed Cross Check= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Avg Speed Cross Check= {mean:.3f} +/- {std:.3f}")

    #print(f"1-Total Mean Avg Speed Cross Check= {np.mean(result_avg_speed_cross_check_list):.3f} +/- {np.std(result_avg_speed_cross_check_list):.3f}")
    #print(f"2-Total Mean Avg Speed Cross Check= {np.mean(result_avg_speed_cross_check_list):.3f} +/- {np.mean(std_avg_speed_cross_check_list):.3f}")
    logging.info(f"1-Total Mean Avg Speed Cross Check= {np.mean(result_avg_speed_cross_check_list):.3f} +/- {np.std(result_avg_speed_cross_check_list):.3f}")
    logging.info(f"2-Total Mean Avg Speed Cross Check= {np.mean(result_avg_speed_cross_check_list):.3f} +/- {np.mean(std_avg_speed_cross_check_list):.3f}")

    ########
    # Avg CO2 Emission TL
    result_average_co2_emission_tl_list = []
    std_average_co2_emission_tl_list = []
    for index, elem in enumerate(all_per_episode_average_co2_emission_tl):
        mean = np.mean(elem)
        std = np.std(elem)
        result_average_co2_emission_tl_list.append(mean)
        std_average_co2_emission_tl_list.append(std)
        # print(f"Episode:{index} Mean Average CO2 Emission TL= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Average CO2 Emission TL= {mean:.3f} +/- {std:.3f}")

    # print(f"1-Total Mean Average CO2 Emission TL= {np.mean(result_average_co2_emission_tl_list):.3f} +/- {np.std(result_average_co2_emission_tl_list):.3f}")
    # print(f"2-Total Mean Average CO2 Emission TL= {np.mean(result_average_co2_emission_tl_list):.3f} +/- {np.mean(std_average_co2_emission_tl_list):.3f}")
    logging.info(f"1-Total Mean Average CO2 Emission TL= {np.mean(result_average_co2_emission_tl_list):.3f} +/- {np.std(result_average_co2_emission_tl_list):.3f}")
    logging.info(f"2-Total Mean Average CO2 Emission TL= {np.mean(result_average_co2_emission_tl_list):.3f} +/- {np.mean(std_average_co2_emission_tl_list):.3f}")

    # Avg CO Emission TL
    result_average_co_emission_tl_list = []
    std_average_co_emission_tl_list = []
    for index, elem in enumerate(all_per_episode_average_co_emission_tl):
        mean = np.mean(elem)
        std = np.std(elem)
        result_average_co_emission_tl_list.append(mean)
        std_average_co_emission_tl_list.append(std)
        # print(f"Episode:{index} Mean Average CO Emission TL= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Average CO Emission TL= {mean:.3f} +/- {std:.3f}")

    # print(f"1-Total Mean Average CO Emission TL= {np.mean(result_average_co_emission_tl_list):.3f} +/- {np.std(result_average_co_emission_tl_list):.3f}")
    # print(f"2-Total Mean Average CO Emission TL= {np.mean(result_average_co_emission_tl_list):.3f} +/- {np.mean(std_average_co_emission_tl_list):.3f}")
    logging.info(f"1-Total Mean Average CO Emission TL= {np.mean(result_average_co_emission_tl_list):.3f} +/- {np.std(result_average_co_emission_tl_list):.3f}")
    logging.info(f"2-Total Mean Average CO Emission TL= {np.mean(result_average_co_emission_tl_list):.3f} +/- {np.mean(std_average_co_emission_tl_list):.3f}")

    # Avg Fuel Consumption TL
    result_average_fuel_consumption_tl_list = []
    std_average_fuel_consumption_tl_list = []
    for index, elem in enumerate(all_per_episode_average_fuel_consumption_tl):
        mean = np.mean(elem)
        std = np.std(elem)
        result_average_fuel_consumption_tl_list.append(mean)
        std_average_fuel_consumption_tl_list.append(std)
        # print(f"Episode:{index} Mean Average Fuel Consumption TL= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Average Fuel Consumption TL= {mean:.3f} +/- {std:.3f}")

    # print(f"1-Total Mean Average Fuel Consumption TL= {np.mean(result_average_fuel_consumption_tl_list):.3f} +/- {np.std(result_average_fuel_consumption_tl_list):.3f}")
    # print(f"2-Total Mean Average Fuel Consumption TL= {np.mean(result_average_fuel_consumption_tl_list):.3f} +/- {np.mean(std_average_fuel_consumption_tl_list):.3f}")
    logging.info(f"1-Total Mean Average Fuel Consumption TL= {np.mean(result_average_fuel_consumption_tl_list):.3f} +/- {np.std(result_average_fuel_consumption_tl_list):.3f}")
    logging.info(f"2-Total Mean Average Fuel Consumption TL= {np.mean(result_average_fuel_consumption_tl_list):.3f} +/- {np.mean(std_average_fuel_consumption_tl_list):.3f}")

    # Avg Noise Emission TL
    result_average_noise_emission_tl_list = []
    std_average_noise_emission_tl_list = []
    for index, elem in enumerate(all_per_episode_average_noise_emission_tl):
        mean = np.mean(elem)
        std = np.std(elem)
        result_average_noise_emission_tl_list.append(mean)
        std_average_noise_emission_tl_list.append(std)
        # print(f"Episode:{index} Mean Average Noise Emission TL= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Average Noise Emission TL= {mean:.3f} +/- {std:.3f}")

    # print(f"1-Total Mean Average Noise Emission TL= {np.mean(result_average_noise_emission_tl_list):.3f} +/- {np.std(result_average_noise_emission_tl_list):.3f}")
    # print(f"2-Total Mean Average Noise Emission TL= {np.mean(result_average_noise_emission_tl_list):.3f} +/- {np.mean(std_average_noise_emission_tl_list):.3f}")
    logging.info(f"1-Total Mean Average Noise Emission TL= {np.mean(result_average_noise_emission_tl_list):.3f} +/- {np.std(result_average_noise_emission_tl_list):.3f}")
    logging.info(f"2-Total Mean Average Noise Emission TL= {np.mean(result_average_noise_emission_tl_list):.3f} +/- {np.mean(std_average_noise_emission_tl_list):.3f}")

    # Avg CO2 Emission
    result_average_co2_emission_list = []
    std_average_co2_emission_list = []
    for index, elem in enumerate(all_per_episode_average_co2_emission):
        mean = np.mean(elem)
        std = np.std(elem)
        result_average_co2_emission_list.append(mean)
        std_average_co2_emission_list.append(std)
        # print(f"Episode:{index} Mean Average CO2 Emission= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Average CO2 Emission= {mean:.3f} +/- {std:.3f}")

    # print(f"1-Total Mean Average CO2 Emission= {np.mean(result_average_co2_emission_list):.3f} +/- {np.std(result_average_co2_emission_list):.3f}")
    # print(f"2-Total Mean Average CO2 Emission= {np.mean(result_average_co2_emission_list):.3f} +/- {np.mean(std_average_co2_emission_list):.3f}")
    logging.info(f"1-Total Mean Average CO2 Emission= {np.mean(result_average_co2_emission_list):.3f} +/- {np.std(result_average_co2_emission_list):.3f}")
    logging.info(f"2-Total Mean Average CO2 Emission= {np.mean(result_average_co2_emission_list):.3f} +/- {np.mean(std_average_co2_emission_list):.3f}")

    # Avg CO Emission
    result_average_co_emission_list = []
    std_average_co_emission_list = []
    for index, elem in enumerate(all_per_episode_average_co_emission):
        mean = np.mean(elem)
        std = np.std(elem)
        result_average_co_emission_list.append(mean)
        std_average_co_emission_list.append(std)
        # print(f"Episode:{index} Mean Average CO Emission= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Average CO Emission= {mean:.3f} +/- {std:.3f}")

    # print(f"1-Total Mean Average CO Emission= {np.mean(result_average_co_emission_list):.3f} +/- {np.std(result_average_co_emission_list):.3f}")
    # print(f"2-Total Mean Average CO Emission= {np.mean(result_average_co_emission_list):.3f} +/- {np.mean(std_average_co_emission_list):.3f}")
    logging.info(f"1-Total Mean Average CO Emission= {np.mean(result_average_co_emission_list):.3f} +/- {np.std(result_average_co_emission_list):.3f}")
    logging.info(f"2-Total Mean Average CO Emission= {np.mean(result_average_co_emission_list):.3f} +/- {np.mean(std_average_co_emission_list):.3f}")

    # Avg Fuel Consumption
    result_average_fuel_consumption_list = []
    std_average_fuel_consumption_list = []
    for index, elem in enumerate(all_per_episode_average_fuel_consumptions):
        mean = np.mean(elem)
        std = np.std(elem)
        result_average_fuel_consumption_list.append(mean)
        std_average_fuel_consumption_list.append(std)
        # print(f"Episode:{index} Mean Average Fuel Consumption= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Average Fuel Consumption= {mean:.3f} +/- {std:.3f}")

    # print(f"1-Total Mean Average Fuel Consumption= {np.mean(result_average_fuel_consumption_list):.3f} +/- {np.std(result_average_fuel_consumption_list):.3f}")
    # print(f"2-Total Mean Average Fuel Consumption= {np.mean(result_average_fuel_consumption_list):.3f} +/- {np.mean(std_average_fuel_consumption_list):.3f}")
    logging.info(f"1-Total Mean Average Fuel Consumption= {np.mean(result_average_fuel_consumption_list):.3f} +/- {np.std(result_average_fuel_consumption_list):.3f}")
    logging.info(f"2-Total Mean Average Fuel Consumption= {np.mean(result_average_fuel_consumption_list):.3f} +/- {np.mean(std_average_fuel_consumption_list):.3f}")

    # Avg Noise Emission
    result_average_noise_emission_list = []
    std_average_noise_emission_list = []
    for index, elem in enumerate(all_per_episode_average_noise_emissions):
        mean = np.mean(elem)
        std = np.std(elem)
        result_average_noise_emission_list.append(mean)
        std_average_noise_emission_list.append(std)
        # print(f"Episode:{index} Mean Average Noise Emission= {mean:.3f} +/- {std:.3f}")
        logging.info(f"Episode:{index} Mean Average Noise Emission= {mean:.3f} +/- {std:.3f}")

    # print(f"1-Total Mean Average Noise Emission= {np.mean(result_average_noise_emission_list):.3f} +/- {np.std(result_average_noise_emission_list):.3f}")
    # print(f"2-Total Mean Average Noise Emission= {np.mean(result_average_noise_emission_list):.3f} +/- {np.mean(std_average_noise_emission_list):.3f}")
    logging.info(f"1-Total Mean Average Noise Emission= {np.mean(result_average_noise_emission_list):.3f} +/- {np.std(result_average_noise_emission_list):.3f}")
    logging.info(f"2-Total Mean Average Noise Emission= {np.mean(result_average_noise_emission_list):.3f} +/- {np.mean(std_average_noise_emission_list):.3f}")
    ####

    #print("****************************************")
    logging.info("****************************************")

    '''
        with open(args.csv_file, 'a') as file:
            for m in metrics:
                file.write(str(m) + ', ')
            file.write('\n')
        '''

    csv_output_writer.write_final_values(args.csv_file, metrics)

    ##draw2(list(), episode_rewards, chart_name='METRIC: Testing Episode Rewards', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_stopped_list, chart_name='METRIC: Testing Episode J0 Stopped', y_label="# of cars", draw_avg=False)
    ##draw2(list(), J0_stopped_cross_check_list, chart_name='METRIC: Testing Episode J0 Stopped', y_label="# of cars", draw_avg=False)
    ##draw2(list(), J0_acc_wt_list, chart_name='METRIC: Testing Episode J0 Acc. Waiting Time', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_avg_speed_list, chart_name='METRIC: Testing Episode J0 Avg. Speed', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_avg_speed_2_list, chart_name='METRIC: Testing Episode J0 Avg. Speed 2', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_avg_speed_cross_check_list, chart_name='METRIC: Testing Episode J0 Avg. Speed Cross Check', y_label="Seconds", draw_avg=False)

    ##draw2(list(), J0_average_co2_emission_tl_list, chart_name='METRIC: Testing Episode J0 Avg. CO2 Emission TL', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_average_co_emission_tl_list, chart_name='METRIC: Testing Episode J0 Avg. CO Emission TL', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_average_fuel_consumption_tl_list, chart_name='METRIC: Testing Episode J0 Avg. Fuel Consumption TL', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_average_noise_emission_tl_list, chart_name='METRIC: Testing Episode J0 Avg. Noise Emission TL', y_label="Seconds", draw_avg=False)

    ##draw2(list(), J0_average_co2_emission_list, chart_name='METRIC: Testing Episode J0 Avg. CO2 Emission', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_average_co_emission_list, chart_name='METRIC: Testing Episode J0 Avg. CO Emission', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_average_fuel_consumption_list, chart_name='METRIC: Testing Episode J0 Avg. Fuel Consumption', y_label="Seconds", draw_avg=False)
    ##draw2(list(), J0_average_noise_emission_list, chart_name='METRIC: Testing Episode J0 Avg. Noise Emission', y_label="Seconds", draw_avg=False)
