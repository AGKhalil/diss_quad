# Train an agent from scratch with PPO2 and save package and learning graphs
# from OpenGL import GLU
import os
import time
import subprocess
import shutil
import gym
import gym_real
import gym_cl
import numpy as np
import matplotlib.pyplot as plt
import datetime
import imageio
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import xml.etree.ElementTree as ET

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model_prof.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, model_name, plt_dir, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    m_name_csv = model_name + ".csv"
    old_file_name = os.path.join(log_folder, "monitor.csv")
    new_file_name = os.path.join(log_folder, m_name_csv)
    save_name = os.path.join(plt_dir, model_name)

    x, y = ts2xy(load_results(log_folder), 'timesteps')
    shutil.copy(old_file_name, new_file_name)
    y = moving_average(y, window=1)
    # Truncate x
    x = x[lfen(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    print('Saving plot at:', save_name)
    plt.savefig(save_name + ".png")
    plt.savefig(save_name + ".eps")
    print("plots saved...")

models_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "prof/models/")
models_tmp_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "prof/models_tmp/")
log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prof/tmp")
gif_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prof/tmp_gif/")
plt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prof/plot")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(gif_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(models_tmp_dir, exist_ok=True)
os.makedirs(plt_dir, exist_ok=True)

step_total = 1000
env_name = 'CurriculumLearning-v0'
save_path = os.path.dirname(
    os.path.realpath(__file__))
env = gym.make(env_name)
env.save_path = save_path
n_cpu = 8
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=step_total)

stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
model_name = "Prof" + "_" + \
    str(step_total) + "_" + stamp
model_loc = os.path.join(models_dir, model_name)
model.save(model_loc)

plot_results(log_dir, model_name, plt_dir)