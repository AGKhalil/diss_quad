import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import shelve
from collections import defaultdict
import cloudpickle

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
    checkpoint_log = run_path + '/checkpoint'
    checkpoint_keeper = []
    with open(checkpoint_log) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                checkpoint_keeper.append(row[0])
            line_count += 1
    return run_path + '/' + checkpoint_keeper[checkpoint] + '.csv'

def load_logger_chkpt(run_path, checkpoint=-1):
    checkpoint_log = run_path + '/checkpoint'
    checkpoint_keeper = []
    with open(checkpoint_log) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            checkpoint_keeper.append(row[0])
    return run_path + checkpoint_keeper[-1]

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

def plot_graph(plot_name, save_name, length, reward, divider):
    reward = moving_average(reward, window=30)
    # Truncate length
    length = length[len(length) - len(reward):]
    title = plot_name
    fig = plt.figure(title)
    plt.plot(length, reward)
    for i in divider:
        plt.axvline(x=i, linestyle='--', color='#ccc5c6', label='leg increment')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.ylim([0, 800])
    plt.savefig(save_name + ".png")
    plt.savefig(save_name + ".eps")
    # plt.show()
    plt.close()

def aggregate_results(length, agg_reward, divider):
    max_r, min_r, avg_r = [], [], []
    # print(len(agg_reward))
    # print([len(i) for i in agg_reward])
    data_length = min([len(i) for i in agg_reward])
    agg_reward = [i[:data_length] for i in agg_reward]
    # print([len(i) for i in agg_reward])
    # print(data_length)
    np.asarray(agg_reward)
    max_r = moving_average(np.max(agg_reward, 0), window=30)
    min_r = moving_average(np.min(agg_reward, 0), window=30)
    avg_r = moving_average(np.mean(agg_reward, axis=0), window=30)
        
    # Truncate length
    length = length[len(length) - len(avg_r):]
    return max_r, min_r, avg_r, length, divider

def plot_aggregate(plot_name, save_name, lengths, agg_rewards, dividers, labels):
    title = plot_name
    colors = ['#7ebcff', '#ffcccb']
    fig = plt.figure(title)
    for i in range(len(lengths)):
        max_r, min_r, avg_r, length, divider = aggregate_results(lengths[i], agg_rewards[i], dividers[i])
        for j in divider:
            plt.axvline(x=j, linestyle='--', color='#ccc5c6')
        plt.plot(length, avg_r, label=labels[i])
        plt.fill_between(length, min_r, max_r, color=colors[i], alpha=0.4)
    plt.xlabel('Number of Timesteps')
    plt.legend()
    plt.ylabel('Rewards')
    plt.title(title)
    plt.ylim([0, 800])
    plt.xlim([0, 5500000])
    plt.savefig(save_name + ".png")
    plt.savefig(save_name + ".eps")
    # plt.show()
    plt.close()

plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/brl/Seagate Expansion Drive1/khalil/plot_saves/")
plot_agg_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/brl/Seagate Expansion Drive1/khalil/plot_saves/agg/")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(plot_agg_dir, exist_ok=True)

exp_logger = '/media/brl/Seagate Expansion Drive1/khalil/tf_save/exp_log/'
exp_file = load_logger_chkpt(exp_logger)
with open(exp_file, "rb") as file:
    data = cloudpickle.load(file)
# print(data['LEG_LENGTH', (-0.1, -0.1)])

# file = shelve.open(exp_logger)
# keys = list(file.keys())
# keys.sort()
# data = defaultdict(list)
# for exp in keys:
#     data[file[exp][0], file[exp][1]].append(file[exp][2])
#     model_names = file[exp][2]
#     # print(exp, file[exp][0], file[exp][1], file[exp][2])
#     length, reward = [], []
#     cum_l = 0
#     cum_l_list = []
#     plot_name = exp + file[exp][0] + str(file[exp][1])
#     save_name = plot_dir + file[exp][0] + str(file[exp][1]) + str(file[exp][2])
#     for model_name in model_names:
#         run_path = '/media/brl/Seagate Expansion Drive1/khalil/tf_save/' + model_name
#         l, r = get_plot_csv(load_checkpoint(run_path))
#         length.extend(l + cum_l)
#         reward.extend(r)
#         # print(model_name, len(r), len(l), l[-1])
#         cum_l += l[-1]
#         cum_l_list.append(cum_l)
#     plot_graph(plot_name, save_name, length, reward, cum_l_list)
# file.close()
# print(data['LEG_LENGTH', (-0.1, -0.1)])

for key, val in data.items():
    lengths, agg_rewards, cum_l_lists = [], [], []
    plot_name = key[0] + '(' + '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ')'
    save_name = plot_agg_dir + plot_name + 'agg'
    if key[1][0] != key[1][-1]:
        values = [val[:5], data[key[0], (key[1][-1], key[1][-1])][:5]]
        labels = ['(' + '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ') curriculum', '(' + '%.1f' % abs(key[1][-1]) + ',' + '%.1f' % abs(key[1][-1]) + ') baseline']
    else:
        values = [val[:5]]
        labels = ['(' + '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ') baseline']
    for i in values:
        agg_reward = []
        for model_names in i:
            length, reward = [], []
            cum_l = 0
            cum_l_list = []
            for model_name in model_names:
                run_path = '/media/brl/Seagate Expansion Drive1/khalil/tf_save/' + model_name
                l, r = get_plot_csv(load_checkpoint(run_path))
                length.extend(l + cum_l)
                reward.extend(r)
                cum_l += l[-1]
                cum_l_list.append(cum_l)
            agg_reward.append(reward)
        agg_rewards.append(agg_reward)
        lengths.append(length)
        cum_l_lists.append(cum_l_list)
    plot_aggregate(plot_name + 'agg' + str(len(val)), save_name, lengths, agg_rewards, cum_l_lists, labels)