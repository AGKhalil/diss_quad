import os
import sys
import gym
import gym_real
import numpy as np
import matplotlib.pyplot as plt
import datetime
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

if __name__ == "__main__":
	file_name = str(sys.argv[1])

	if file_name[:3] == "mod":
		model_name = file_name
	else:
		dirpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
		log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
		model_name = os.path.join(dirpath, file_name)

	env_name = 'Real-v0'
	env = gym.make(env_name)
	model = PPO2.load(model_name)

	obs = env.reset()
	for i in range(1000):
	  action, _states = model.predict(obs)
	  obs, rewards, dones, info = env.step(action)
	  env.render()