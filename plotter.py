"""Script written to generate results from the training log files. The results generated are plots and gifs.

Attributes:
    baselines_agg (bool): flag to aggregate results for baseline plots
    counter (int): used for plotting baselines of list A and B seprately
    exp_file (str): logger file path
    exp_logger (str): logger directory path
    leg_lengths_1 (int): list A curriculum leg lengths
    leg_lengths_2 (int): list B curriculum leg lengths
    names (list): model names
    plot_agg_dir (str): plots aggregate directory
    plot_dir (str): plots directory
"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations
import shelve
from collections import defaultdict
import cloudpickle
import gym
import gym_real
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import imageio

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    
    Args:
        values (int): list of values to do moving average for
        window (int): window of moving average
    
    Returns:
        int: new list of moving average applied to the input
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def load_checkpoint(run_path, checkpoint=-1, csv_ind=True):
    """Loads a model's last checkpoint csv or pkl file
    
    Args:
        run_path (str): model path
        checkpoint (str, last checpoint by default): checkpoint to load
        csv_ind (bool, loads csv by default): Description
    
    Returns:
        str: path to correct file
    """
    checkpoint_log = run_path + '/checkpoint'
    checkpoint_keeper = []
    with open(checkpoint_log) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                checkpoint_keeper.append(row[0])
            line_count += 1
    if csv_ind:
        return run_path + '/' + checkpoint_keeper[checkpoint] + '.csv'
    else:
        return run_path + '/' + checkpoint_keeper[checkpoint] + '.pkl'

def load_logger_chkpt(run_path, checkpoint=-1):
    """Loads most recent experiment log file
    
    Args:
        run_path (str): file path
        checkpoint (int, last checkpoint by default): checkpoint number
    
    Returns:
        str: path to correct log file at the needed checkpoint
    """
    checkpoint_log = run_path + '/checkpoint_1'
    checkpoint_keeper = []
    with open(checkpoint_log) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            checkpoint_keeper.append(row[0])
    return run_path + checkpoint_keeper[-1]

def get_plot_csv(plot_log):
    """Extract results from csv file
    
    Args:
        plot_log (str): csv file path
    
    Returns:
        int, int: length, reward
    """
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
    """Plots a graph of reward vs time
    
    Args:
        plot_name (str): name of plot
        save_name (str): save name of plot
        length (int): list of time steps
        reward (int): list of rewards
        divider (int): timestep of task switch
    """
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
    plt.close()

def aggregate_results(length, agg_reward, divider):
    """Aggregates the results of seeds for an experiment
    
    Args:
        length (int): time steps of results
        agg_reward (int): list of rewards per seed
        divider (int): list of timesteps at task swtich
    
    Returns:
        int: aggregate data needed to plot results
    """
    max_r, min_r, avg_r = [], [], []
    data_length = min([len(i) for i in agg_reward])
    agg_reward = [i[:data_length] for i in agg_reward]
    np.asarray(agg_reward)
    max_r = moving_average(np.max(agg_reward, 0), window=30)
    min_r = moving_average(np.min(agg_reward, 0), window=30)
    avg_r = moving_average(np.mean(agg_reward, axis=0), window=30)
    std = moving_average(np.std(agg_reward, axis=0), window=30)
        
    # Truncate length
    length = length[len(length) - len(avg_r):]
    return max_r, min_r, avg_r, std, length, divider

def plot_aggregate(plot_name, save_name, lengths, agg_rewards, dividers, labels):
    """Plots the aggregate cumulative reward vs time of all seeds of an experiment.
    
    Args:
        plot_name (str): name of plot
        save_name (str): save name of plot
        lengths (int): list of time steps per seed
        agg_rewards (int): list of rewards per seed
        dividers (int): list of dividers per seed
        labels (str): list of model names
    """
    title = plot_name[0] + plot_name[1]
    widths = [i * 0.5 for i in range(2, 5)] + [i * 0.2 for i in range(5, 9)]
    alphas = [i * 0.1 for i in range(8)]
    colors = ['#db4437', '#4285f4', '#f4b400', '#0f9d58', '#00796b', '#5c6bc0', '#00acc1', '#ff7043']
    fig = plt.figure(title)
    for i in range(len(lengths)):
        max_r, min_r, avg_r, std, length, divider = aggregate_results(lengths[i], agg_rewards[i], dividers[i])
        plt.plot(length, avg_r, label=labels[i], color=colors[i], linewidth=widths[i])
        plt.fill_between(length, min_r, max_r, color=colors[i], alpha=0.1)
    for i, j in enumerate(divider):
        if i == 0:
            plt.axvline(x=j, linestyle=':', color='#000000', label='end of training instance')
        else:
            plt.axvline(x=j, linestyle=':', color='#000000')
    plt.xlabel('Timesteps')
    plt.legend()
    plt.ylabel('Cumulative Reward')
    plt.title(title)
    plt.ylim([0, 800])
    plt.xlim([0, 5500000])
    plt.savefig(save_name + ".png")
    plt.savefig(save_name + ".pdf")
    plt.close()
    std_fig = plt.figure()
    for i in range(len(lengths)):
        max_r, min_r, avg_r, std, length, divider = aggregate_results(lengths[i], agg_rewards[i], dividers[i])
        plt.plot(length, std, label=labels[i], color=colors[i], linewidth=widths[i])
    for i, j in enumerate(divider):
        if i == 0:
            plt.axvline(x=j, linestyle=':', color='#000000', label='end of training instance')
        else:
            plt.axvline(x=j, linestyle=':', color='#000000')
    plt.xlabel('Timesteps')
    plt.legend()
    plt.ylabel('Running Standard Deviation')
    plt.title(plot_name[0] + ' Standard Deviation')
    plt.ylim([0, 300])
    plt.xlim([0, 5500000])
    plt.savefig(save_name + "std.png")
    plt.savefig(save_name + "std.pdf")
    plt.close()

def make_me_a_gif(load_name, model_name):
    """Creates a gif of the agent.
    
    Args:
        load_name (str): load pkl name
        model_name (str): model name
    """
    loader = load_checkpoint('/media/brl/Seagate Expansion Drive1/khalil/tf_save/' + load_name, checkpoint=-1, csv_ind=False)
    gif_dir = '/media/brl/Seagate Expansion Drive1/khalil/report_gifs/'
    os.makedirs(gif_dir, exist_ok=True)
    save_str = gif_dir + model_name + '.gif'
    images = []
    env = gym.make('Real-v0')
    model = PPO2.load(loader)
    obs = env.reset()
    done = False
    step, reward = 0, 0
    img = env.sim.render(
        width=300, height=300)
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        img = env.sim.render(
            width=300, height=300, camera_name='isometric_view') #camera_name="isometric_view"
        images.append(np.flipud(img))
        step += 1

    env.close()
    del env
    del model

    print("creating gif...")
    print("saving gif at:", save_str)
    imageio.mimsave(save_str, [np.array(img)
                               for i, img in enumerate(images) if i % 2 == 0], fps=29)
    print("gif created...")

plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/brl/Seagate Expansion Drive1/khalil/plot_saves/")
plot_agg_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/brl/Seagate Expansion Drive1/khalil/plot_saves/agg/")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(plot_agg_dir, exist_ok=True)

exp_logger = '/media/brl/Seagate Expansion Drive1/khalil/tf_save/exp_log/'
exp_file = load_logger_chkpt(exp_logger)
with open(exp_file, "rb") as file:
    data = cloudpickle.load(file)

baselines_agg = True
leg_lengths_1 = [i * -0.1 for i in range(1, 5)]
leg_lengths_2 = [i * -0.1 for i in range(1, 9, 2)]
counter = 0
for key, val in data.items():
    lengths, agg_rewards, cum_l_lists = [], [], []
    if key[1][0] != key[1][-1]:
        plot_name = ['Leg lengths: (' + '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ') ', 'Learning Curves']
        save_name = plot_agg_dir + '(' +  '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ')' + 'agg'
        values = [val[:5], data[key[0], (key[1][-1], key[1][-1])][:5]]
        labels = ['(' + '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ') curriculum', '(' + '%.1f' % abs(key[1][-1]) + ',' + '%.1f' % abs(key[1][-1]) + ') tabula rasa']
    elif key[1][0] == key[1][-1] and baselines_agg:
        if counter == 0:
            plot_name = ['List A Baseline ', 'Learning Curves']
            save_name = plot_agg_dir + 'baselines_aggA'
            base_length = leg_lengths_1
            counter += 1
        else:
            plot_name = ['List B Baseline', 'Learning Curves']
            save_name = plot_agg_dir + 'baselines_aggB'
            base_length = leg_lengths_2
            baselines_agg = False
        values = [data[key[0], (i, i)] for i in base_length]
        labels = ['(' + '%.1f' % abs(i) + ',' + '%.1f' % abs(i) + ')' for i in base_length]
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
    plot_aggregate(plot_name, save_name, lengths, agg_rewards, cum_l_lists, labels)

names = ['listA', 'listB']
for v, variant in enumerate([leg_lengths_1, leg_lengths_2]):
    perm_vars = list(permutations(variant, 2))
    for var in variant:
        lengths, agg_rewards, cum_l_lists = [], [], []
        kys = [j for j in perm_vars if var == j[-1]] + [(var, var)]
        values = [data['LEG_LENGTH', ky][:5] for ky in kys]
        plot_name = ['Final leg length: (' + '%.1f' % abs(var) + ') ', 'Learning Curves']
        save_name = plot_agg_dir + names[v] + '(' + '%.1f' % abs(var) + ')' + 'agg'
        labels = ['(' + '%.1f' % abs(ky[0]) + ',' + '%.1f' % abs(ky[-1]) + ') curriculum' for ky in kys[:-1]] + ['(' + '%.1f' % abs(ky[0]) + ',' + '%.1f' % abs(ky[-1]) + ') tabula rasa' for ky in kys[-1:]]
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
        plot_aggregate(plot_name, save_name, lengths, agg_rewards, cum_l_lists, labels)

for v, variant in enumerate([leg_lengths_1, leg_lengths_2]):
    perm_vars = list(permutations(variant, 2))
    for var in variant:
        lengths, agg_rewards, cum_l_lists = [], [], []
        kys = [j for j in perm_vars if var == j[0]] + [(var, var)]
        values = [data['LEG_LENGTH', ky][:5] for ky in kys]
        plot_name = ['Initial leg length: (' + '%.1f' % abs(var) + ') ', 'Learning Curves']
        save_name = plot_agg_dir + names[v] + 'Initial(' + '%.1f' % abs(var) + ')' + 'agg'
        labels = ['(' + '%.1f' % abs(ky[0]) + ',' + '%.1f' % abs(ky[-1]) + ') curriculum' for ky in kys[:-1]] + ['(' + '%.1f' % abs(ky[0]) + ',' + '%.1f' % abs(ky[-1]) + ') tabula rasa' for ky in kys[-1:]]
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
        plot_aggregate(plot_name, save_name, lengths, agg_rewards, cum_l_lists, labels)

for key, val in data.items():
    lengths, agg_rewards, cum_l_lists = [], [], []
    if key[1][0] != key[1][-1]:
        plot_name = ['Leg lengths: (' + '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ') and (' + '%.1f' % abs(key[1][-1]) + ',' + '%.1f' % abs(key[1][0]) + ')', ' Learning Curves']
        save_name = plot_agg_dir + '(' +  '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ')' + 'viceversa'
        values = [val[:5], data[key[0], (key[1][-1], key[1][0])][:5]]
        labels = ['(' + '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ') curriculum', '(' + '%.1f' % abs(key[1][-1]) + ',' + '%.1f' % abs(key[1][0]) + ') curriculum']
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
    plot_aggregate(plot_name, save_name, lengths, agg_rewards, cum_l_lists, labels)

for key, val in data.items():
    for v in val:
        gif_name = '(' +  '%.1f' % abs(key[1][0]) + ',' + '%.1f' % abs(key[1][-1]) + ')' + v[-1]
        make_me_a_gif(v[-1], gif_name)