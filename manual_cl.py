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
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.tf_util import save_state
from stable_baselines.common import pydrive_util
import xml.etree.ElementTree as ET


def alter_env(exp_type, variant):
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
    checkpoint_keeper = []
    with open('tmp/tmp_file') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            checkpoint_keeper.append(row[0])
    return run_path + '/' + checkpoint_keeper[checkpoint] + '.pkl'


def log_experiments(exp_num, exp_type, variants, model_names, exp_log, log_dict, drive):
    file = shelve.open(exp_log)
    file['exp' + exp_num] = [exp_type, variants, model_names]
    file.close()
    print('exp' + exp_num)
    pydrive_util.upload_file(drive, 'tf_save/' + exp_log)
    # with open(exp_log, 'a') as csv_file:
    # 	csv_writer = csv.writer(csv_file, delimiter=',')
    # 	if os.stat(exp_log).st_size == 0:
    # 		csv_writer.writerow(['exp', 'type', 'variants', 'models'])
    # 	csv_writer.writerow(['exp' + str(exp_num), exp_type, variants, model_names])


def run_experiment(exp_num, exp_type, variants, n_cpu, step_total, exp_log, log_dict, drive, og_dir):
    model_names = []
    run_path = ''
    for order, variant in enumerate(variants):
        alter_env(exp_type, variant)
        env = gym.make("Real-v0")
        env = Monitor(env, 'tf_save', allow_early_resets=True)
        env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
        if order == 0:
            model = PPO2(MlpPolicy, env, verbose=0,
                         tensorboard_log="./tensorboard_log/", drive=drive, og_dir=og_dir)
        else:
            pydrive_util.download_file(drive, run_path + '/checkpoint')
            load_name = load_checkpoint(-1, run_path)
            pydrive_util.download_file(drive, load_name)
            model = PPO2.load('tmp/tmp_file', env=env, drive=drive, og_dir=og_dir)
        model_names.append(model.model_name)
        run_path = model.graph_dir
        model.learn(total_timesteps=step_total)
        pydrive_util.upload_file(drive, model.checkpoint_log)
        env.close()
        del model, env
    log_experiments(exp_num, exp_type, variants,
                    model_names, exp_log, log_dict, drive)

if __name__ == "__main__":
    n_cpu = 8
    n_step = 128
    desired_log_pts = 10
    step_total = desired_log_pts * n_cpu * n_step
    leg_lengths = [i * -0.1 for i in range(1, 5)]
    goal_diss = [i * -2 for i in range(2, 6)]

    drive = pydrive_util.drive_auth()
    og_dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(og_dir, 'tmp'), exist_ok=True)

    exp_log = 'experiment_logs'
    log_dict = {}
    model_names = []

    leg_type = 'LEG_LENGTH'
    dis_type = 'GOAL_DIS'
    exp_types = [leg_type, dis_type]
    for i in range(5):
        for j, exp_type in enumerate(exp_types):
            if exp_type == leg_type:
                variant = leg_lengths
            elif exp_type == dis_type:
                variant = goal_diss
            perm_2 = list(permutations(variant, 2))
            perm_4 = list(permutations(variant))
            perms = perm_2 + perm_4
            for k, perm in enumerate(perms):
                run_experiment(str(i) + '_' + str(j) + '_' + str(
                    k), exp_type, perm, n_cpu, step_total, exp_log, log_dict, drive, og_dir)
                pydrive_util.clean_up('tf_save')
