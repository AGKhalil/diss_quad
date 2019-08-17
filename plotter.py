import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import shelve
from stable_baselines.common import pydrive_util

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def load_checkpoint(run_path, checkpoint=-1):
    checkpoint_keeper = []
    with open('tmp/tmp_plt_file') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            checkpoint_keeper.append(row[0])
    return run_path + '/' + checkpoint_keeper[checkpoint] + '.csv'

def get_plot_csv(plot_log):
    reward, length, time = [], [], []
    with open(plot_log) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                reward.append(float(row[0]))
                length.append(int(row[1]))
                time.append(float(row[2]))
            line_count += 1

    length = np.cumsum(length)  
    return length, reward

def plot_graph(plot_name, save_name, length, reward):
    reward = moving_average(reward, window=30)
    # Truncate length
    length = length[len(length) - len(reward):]
    title = plot_name
    fig = plt.figure(title)
    plt.plot(length, reward)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.ylim([0, 800])
    plt.savefig(save_name + ".png")
    plt.savefig(save_name + ".eps")
    # plt.show()
    plt.close()

plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot_saves/")
os.makedirs(plot_dir, exist_ok=True)

drive = pydrive_util.drive_auth()

exp_logger = 'tf_save/experiment_logs'
pydrive_util.download_file(drive, exp_logger, plot=True, db=True)

file = shelve.open('tmp/tmp_db_file')
keys = list(file.keys())
keys.sort()
for exp in keys:
    model_names = file[exp][2]
    print(exp, file[exp][2])
    length, reward = [], []
    cum_l = 0
    plot_name = exp + file[exp][0] + str(file[exp][1])
    save_name = plot_dir + file[exp][0] + str(file[exp][1]) + str(file[exp][2])
    for model_name in model_names:
        run_path = 'tf_save/' + model_name
        pydrive_util.download_file(drive, run_path + '/checkpoint', plot=True)
        load_name = load_checkpoint(run_path)
        pydrive_util.download_file(drive, load_name, plot=True)
        l, r = get_plot_csv('tmp/tmp_plt_file')
        length.extend(l + cum_l)
        reward.extend(r)
        cum_l += l[-1]
    plot_graph(plot_name, save_name, length, reward)
file.close()
pydrive_util.clean_up('tmp')
