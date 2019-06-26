# Train an agent from scratch with PPO2 and save package and learning graphs
# from OpenGL import GLU
import os
import time
import subprocess
import shutil
import gym
import gym_real
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

best_mean_reward, n_steps, old_steps, total_gif_time = -np.inf, 0, 0, 0
step_total = 50000

if step_total >= 1000000:
    n_gifs = 5
else:
    n_gifs = 2
log_incs = np.round((step_total / n_gifs) * 60 / 60000)
env_name = 'Real-v0'

##############################################Functions###################


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, old_steps, gif_dir, env_name, log_incs, models_tmp_dir, total_gif_time
    # Print stats every 1000 calls

    if abs(n_steps - old_steps) >= log_incs:
        gif_start = time.time()
        old_steps = n_steps
        # Evaluate policy performance
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
                _locals['self'].save(models_tmp_dir + 'best_model.pkl')

        stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        gif_name = "PPO2_" + env_name + "_" + str(step_total) + "_" + stamp
        save_str = gif_dir + gif_name + '.gif'
        images = []

        env_gif = gym.make(env_name)
        obs = env_gif.reset()
        img = env_gif.sim.render(
            width=200, height=200, camera_name="isometric_view")
        for _ in range(5000):
            action, _ = model.predict(obs)
            obs, _, _, _ = env_gif.step(action)
            img = env_gif.sim.render(
                width=200, height=200, camera_name="isometric_view")
            images.append(np.flipud(img))

        print("creating gif...")
        imageio.mimsave(save_str, [np.array(img)
                                   for i, img in enumerate(images) if i % 2 == 0], fps=29)
        print("gif created...")
        gif_end = time.time()
        total_gif_time += gif_end - gif_start
    n_steps += 1

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
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(save_name + ".png")
    plt.savefig(save_name + ".eps")
    print("plots saved...")
    plt.show()


def alter_leg(leg_length):
    xml_path = os.path.join(gym_real.__path__[0], "envs/assets/real.xml")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    for geom in root.findall("worldbody/body/body/body/body/geom"):
        geom.set("fromto", "0 0 0 0 0 " + str(leg_length))
        print(geom.get("fromto"))

    for pos in root.findall("worldbody/body/[@name='torso']"):
        pos.set("pos", "0 0 " + str(abs(leg_length) + 0.7))
        print(pos.get('pos'))

    tree.write(xml_path)

############################################Traing Models#################

print("running...")
# Create log dir
models_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "models/")
models_tmp_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "models_tmp/")
log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
gif_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp_gif/")
plt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(gif_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(models_tmp_dir, exist_ok=True)
os.makedirs(plt_dir, exist_ok=True)
# Create and wrap the environment
env = gym.make(env_name)
# env = Monitor(env, log_dir, allow_early_resets=True)
# env = DummyVecEnv([lambda: env])

# multiprocess environment
n_cpu = 8
env = Monitor(env, log_dir, allow_early_resets=True)
env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
# Add some param noise for exploration

alter_leg(-5.0)

model = PPO2(MlpPolicy, env, verbose=1)
start = time.time()
model.learn(total_timesteps=step_total, callback=callback)
end = time.time()

# del model

alter_leg(-0.3)

model = PPO2(MlpPolicy, env, verbose=1)
start = time.time()
model.learn(total_timesteps=step_total, callback=callback)
end = time.time()

training_time = end - start - total_gif_time

stamp = ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
model_name = "PPO2_" + env_name + "_" + \
    str(step_total) + "_" + stamp + "_" + str(training_time)
model_loc = os.path.join(models_dir, model_name)
print(model_loc)
model.save(model_loc)

print("Training time:", training_time)
print("model saved as: " + model_name)

plot_results(log_dir, model_name, plt_dir)

del model  # remove to demonstrate saving and loading
env = gym.make(env_name)

# Enjoy trained agent
watch_agent = input("Do you want to watch your sick gaits? (Y/n):")
print("********************************************************************")
print("To keep replaying after the env closes hit ENTER, to quit hit ctrl+c")
print("********************************************************************")
while watch_agent == "y" or "Y":
    subprocess.Popen(
        '''export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so; python load_agent.py '%s' '%s' ''' % (env_name, model_name), shell=True)
    watch_agent = input("Do you want to watch your sick gaits? (Y/n):")
