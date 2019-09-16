"""Summary
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
from itertools import permutations
import imageio
import csv
import shelve
import cloudpickle
from collections import defaultdict
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.tf_util import save_state
import xml.etree.ElementTree as ET


def alter_env(exp_type, variant):
    """Alters leg length, or distance from goal, of the gym-real quadrupedal environment.
    
    Args:
        exp_type (str): leg length or distnace from goal
        variant (float): variable to change
    """
    xml_path = os.path.join(gym_real.__path__[0], "envs/assets/real.xml")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    if exp_type == 'LEG_LENGTH':
        for geom in root.findall("worldbody/body/body/body/body/geom"):
            geom.set("fromto", "0 0 0 0 0 " + str(variant))

        for pos in root.findall("worldbody/body/[@name='torso']"):
            pos.set("pos", "-10.0 0 " + str(abs(variant) + 0.7))
    elif exp_type == 'GOAL_DIS':
        for pos in root.findall("worldbody/body/[@name='torso']"):
            pos.set("pos", str(variant) + " 0 " + str(abs(-0.1) + 0.7))

    tree.write(xml_path)


def load_checkpoint(checkpoint, run_path):
    """Loads a model checkpoint pkl file.
    
    Args:
        checkpoint (int): checkpoint number
        run_path (str): model path
    
    Returns:
        str: path for loading the appropriate model and at at certain checkpoint
    """
    checkpoint_log = run_path + '/checkpoint'
    checkpoint_keeper = []
    with open(checkpoint_log) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            checkpoint_keeper.append(row[0])
    return run_path + '/' + checkpoint_keeper[checkpoint] + '.pkl'


def log_experiments(exp_num, exp_type, variants, model_names, exp_log, log_dict):
    """Records the models run for their corresponding curricula. This allows for easy lookup for results analysis.
    
    Args:
        exp_num (int): experiemnt number
        exp_type (str): type of experiemnt. leg length or goal distance
        variants (int): curriculum list of ints
        model_names (str): list of model names for each part of the curriculum
        exp_log (str): log file path
        log_dict (Dict): dictionary saving logging data
    
    Returns:
        Dict: dictionary saving logging data
    """
    log_dict[exp_type, variants].append(model_names)
    with open(exp_log, "wb") as file_:
        cloudpickle.dump(log_dict, file_)
    print('exp' + exp_num)
    return log_dict


def run_experiment(exp_num, exp_type, variants, n_cpu, step_total, exp_log, log_dict):
    """Runs a curriculum experiment while logging everything.
    
    Args:
        exp_num (int): experiemnt number
        exp_type (str): type of experiemnt. leg length or goal distance
        variants (int): curriculum list of ints
        n_cpu (int): number of CPUs used for training
        step_total (int): total number of training steps
        exp_log (str): log file path
        log_dict (Dict): dictionary saving logging data
    
    Returns:
        Dict: dictionary saving logging data
    """
    model_names = []
    run_path = ''
    for order, variant in enumerate(variants):
        alter_env(exp_type, variant)
        env = gym.make("Real-v0")
        env = Monitor(env, 'tf_save', allow_early_resets=True)
        env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
        if order == 0:
            model = PPO2(MlpPolicy, env, verbose=0,
                         tensorboard_log="./tensorboard_log/")
        else:
            load_name = load_checkpoint(-1, run_path)
            model = PPO2.load(load_name, env=env)
        model_names.append(model.model_name)
        run_path = model.graph_dir
        model.learn(total_timesteps=step_total)
        env.close()
        del model, env
    log_dict = log_experiments(
        exp_num, exp_type, variants, model_names, exp_log, log_dict)
    return log_dict


if __name__ == "__main__":
    n_cpu = 20
    n_step = 128
    desired_log_pts = 1000
    step_total = desired_log_pts * n_cpu * n_step
    leg_lengths_1 = [i * -0.1 for i in range(1, 5)]
    leg_lengths_2 = [i * -0.1 for i in range(1, 9, 2)]
    goal_diss = [i * -2 for i in range(2, 6)]

    log_dir = '/media/brl/Seagate Expansion Drive1/khalil/tf_save/exp_log/'
    os.makedirs(log_dir, exist_ok=True)
    log_dict = defaultdict(list)

    leg_type = 'LEG_LENGTH'
    dis_type = 'GOAL_DIS'
    exp_types = [leg_type]
    lengths = 0
    for exp in [0, 1]:
        for l in range(2, 3):
            for i in range(5):
                for j, exp_type in enumerate(exp_types):
                    if exp_type == leg_type:
                        if exp == 0:
                            variant = leg_lengths_1
                            baseline = [(m,) * l for m in variant]
                        else:
                            variant = leg_lengths_2
                            baseline = [(m,) * l for m in variant[2:]]
                    elif exp_type == dis_type:
                        variant = goal_diss
                    perm_vars = list(permutations(variant, l))
                    perms = baseline + perm_vars
                    for k, perm in enumerate(perms):
                        print(perm)
                        exp_name = '{:%d%m%y_%H:%M:%S}'.format(
                            datetime.datetime.now())
                        exp_log = log_dir + exp_name
                        log_dict = run_experiment(str(
                            i) + '_' + str(j) + '_' + str(k), exp_type, perm, n_cpu, step_total, exp_log, log_dict)
                        checkpoint_log = log_dir + 'checkpoint'
                        print(checkpoint_log)
                        with open(checkpoint_log, mode='a') as employee_file:
                            employee_writer = csv.writer(employee_file)
                            employee_writer.writerow([exp_name])
