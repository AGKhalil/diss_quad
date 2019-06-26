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

env_name = 'CurriculumLearning-v0'
save_path = os.path.dirname(
    os.path.realpath(__file__))
env = gym.make(env_name)
env.save_path = save_path
n_cpu = 8
# env = Monitor(env, log_dir, allow_early_resets=True)
# env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)
