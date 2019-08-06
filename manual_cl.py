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
from itertools import permutations
import imageio
import csv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.tf_util import save_state
import xml.etree.ElementTree as ET


def alter_env(exp_type, variant):
	xml_path = os.path.join(gym_real.__path__[0], "envs/assets/real.xml")
	print(xml_path)

	tree = ET.parse(xml_path)
	root = tree.getroot()
	if exp_type == 'LEG_LENGTH':
		for geom in root.findall("worldbody/body/body/body/body/geom"):
			geom.set("fromto", "0 0 0 0 0 " + str(variant))

		for pos in root.findall("worldbody/body/[@name='torso']"):
			pos.set("pos", "-10.0 0 " + str(abs(variant) + 0.7))
	elif exp_type == 'GOAL_DIS':
		for pos in root.findall("worldbody/body/[@name='torso']"):
			pos.set("pos", str(variant) + " 0 "  + str(abs(-0.1) + 0.7))

	tree.write(xml_path)

def load_checkpoint(model_name, checkpoint, run_path):
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

def log_experiments(exp_num, exp_type, variants, model_names, exp_log):
	log_dict = dict()
	log_dict['exp' + str(exp_num)] = {'type': exp_type, 'variants': variants, 'models': model_names}
	with open(exp_log, 'a') as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerows(log_dict.items())

def run_experiment(exp_num, exp_type, variants, n_cpu, step_total, exp_log):
	model_names = []
	run_path = ''
	for order, variant in enumerate(variants):
		alter_env(exp_type, variant)
		env = gym.make("Real-v0")
		env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
		if order == 0:
			model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard_log/")
		else:
			load_name = load_checkpoint(model_names[-1], -1, run_path)
			model = PPO2.load(load_name, env=env)
		model_names.append(model.model_name)
		run_path = model.graph_dir
		model.learn(total_timesteps=step_total)
		env.close()
		del model, env
	log_experiments(exp_num, exp_type, variants, model_names, exp_log)

if __name__ == "__main__":
	n_cpu = 8
	n_step = 128
	desired_log_pts = 5
	step_total = desired_log_pts * n_cpu * n_step
	leg_lengths = [i * -0.1 for i in range(1, 5)]
	goal_diss = [i * -2 for i in range(2, 6)]

	exp_log = 'experiment_logs.csv'

	leg_type = 'LEG_LENGTH'
	dis_type = 'GOAL_DIS'
	exp_types = [dis_type, leg_type]
	for _ in range(5):
		for exp_type in exp_types:
			if exp_type == leg_type:
				variant = leg_lengths
			elif exp_type == dis_type:
				variant = goal_diss
			perm_2 = list(permutations(variant, 2))
			perm_4 = list(permutations(variant))
			perms = perm_2 + perm_4
			for i, perm in enumerate(perms):
				print('PERM', perms)
				run_experiment(i, exp_type, perm, n_cpu, step_total, exp_log)

