import csv
import matplotlib.pyplot as plt
import numpy as np
import shelve

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def load_checkpoint(checkpoint, run_path):
    checkpoint_log = run_path + '/checkpoint'
    checkpoint_keeper = []
    with open(checkpoint_log) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                checkpoint_keeper.append(row[1])
            line_count += 1
    return run_path + '/' + checkpoint_keeper[checkpoint] + '.pkl'

exp_logger = 'experiment_logs'
# log_dict = {}
file = shelve.open(exp_logger)
keys = list(file.keys())
keys.sort()
for i in keys:
    print(i, file[i])
file.close()
# with open(exp_logger) as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         print(ast.literal_eval(row[3]))

# plot_log = run_path + checkpoint_keeper[-1] + '.csv'
# print(len(checkpoint_keeper))
# reward, length, time = [], [], []
# with open(plot_log) as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#     	if line_count != 0:
#     		reward.append(float(row[0]))
#     		length.append(int(row[1]))
#     		time.append(float(row[2]))
#     	line_count += 1

# length = np.cumsum(length)
# print(len(length))

# # reward = moving_average(reward, window=1)
# # Truncate length
# length = length[len(length) - len(reward):]

# title = 'my_plot'
# fig = plt.figure(title)
# plt.plot(length, reward)
# plt.xlabel('Number of Timesteps')
# plt.ylabel('Rewards')
# # plt.title(title + " Smoothed")
# # plt.savefig(save_name + ".png")
# # plt.savefig(save_name + ".eps")
# # print("plots saved...")
# plt.show()