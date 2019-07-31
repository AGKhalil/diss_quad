"""Script for running the gym-cl environment. This is to test Automatic Curriculum Generation (ACG)

Attributes:
    env (gym.env): a gym environment
    env_name (str): name of the environment
    log_dir (str): training log directory path
    model (BaseRLModel): RL model for training (prof)
    model_loc (str): model save location
    model_name (str): model name
    models_dir (str): model save directory
    n_cpu (int): number of cpus used for multiprocessing
    reset_timesteps (int): max number of steps in a prof episode
    save_path (str): path to parent directory of all data
    stamp (str): time stamp
    step_total (int): total prof training time steps
    worker_total_timesteps (int): total training time steps per worker
"""
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
    if (n_steps + 1) % 10 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model_prof.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True

models_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "models/")
log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prof/tmp")


step_total = 1000
env_name = 'CurriculumLearning-v0'
save_path = os.path.dirname(
    os.path.realpath(__file__))
stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
save_path = save_path
n_cpu = 8
worker_total_timesteps = 100000
reset_timesteps = 10
model_name = "Prof" + "_p" + \
    str(step_total) + '_w' + str(worker_total_timesteps) + "_" + stamp

env = gym.make(env_name, save_path=save_path, n_cpu=n_cpu, worker_total_timesteps=worker_total_timesteps,
               reset_timesteps=reset_timesteps, prof_name=model_name)

env.prof_name = model_name

env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=step_total)

model_loc = os.path.join(models_dir, model_name)
model.save(model_loc)
