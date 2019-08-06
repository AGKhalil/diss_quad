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
from stable_baselines.common.tf_util import save_state
import xml.etree.ElementTree as ET


log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
os.makedirs(log_dir, exist_ok=True)
n_cpu = 8
n_step = 128
step_total = 1000 * n_cpu * n_step

env = gym.make("Real-v0")
env = Monitor(env, log_dir, allow_early_resets=True)
env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard_log/")
print(model.timestamp)
model.learn(total_timesteps=step_total)