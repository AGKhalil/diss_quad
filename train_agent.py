# Train an agent from scratch with PPO2 and save package and learning graphs
# from OpenGL import GLU
import os
import time
import mujoco_py as mj
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

best_mean_reward, n_steps, old_steps = -np.inf, 0, 0
step_total = 100000
env_name = 'Real-v0'

##############################################Functions#################################################

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward, old_steps, gif_dir
  # Print stats every 1000 calls

  if abs(n_steps - old_steps) >= 25:
    old_steps = n_steps
    # Evaluate policy performance
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
        _locals['self'].save(log_dir + 'best_model.pkl')

    stamp =' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    gif_name = "PPO2_"+env_name+"_"+str(step_total)+"_"+stamp
    save_str = gif_dir + gif_name + '.gif'
    images = []
    obs = env.reset()
    img = env.render(mode='rgb_array')
    for _ in range(1000):
      action, _ = model.predict(obs)
      obs, _, _ ,_ = env.step(action)
      img = env.render(mode='rgb_array')
      images.append(img)

    imageio.mimsave(save_str, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
    print("gif created...")
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


def plot_results(log_folder, title='Learning Curve'):
  """
  plot the results

  :param log_folder: (str) the save location of the results to plot
  :param title: (str) the title of the task to plot
  """
  x, y = ts2xy(load_results(log_folder), 'timesteps')
  y = moving_average(y, window=50)
  # Truncate x
  x = x[len(x) - len(y):]

  fig = plt.figure(title)
  plt.plot(x, y)
  plt.xlabel('Number of Timesteps')
  plt.ylabel('Rewards')
  plt.title(title + " Smoothed")
  plt.show()
  plt.waitKey(0)

############################################Traing Models###############################################

print("running...")
# Create log dir
log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
gif_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp_gif")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(gif_dir, exist_ok=True)
# Create and wrap the environment
env = gym.make(env_name)
# env = Monitor(env, log_dir, allow_early_resets=True)
# env = DummyVecEnv([lambda: env])

# multiprocess environment
n_cpu = 20
env = Monitor(env, log_dir, allow_early_resets=True)
env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
# Add some param noise for exploration

model = PPO2(MlpPolicy, env, verbose=1)
start = time.time()
model.learn(total_timesteps=step_total)
end = time.time()

training_time = end - start

stamp =' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
model_name = "PPO2_"+env_name+"_"+str(step_total)+"_"+stamp+"_"+str(training_time)
model.save(model_name)

print("Training time:", training_time)
print("model saved as: " + model_name)

del model # remove to demonstrate saving and loading
env = gym.make(env_name)
model = PPO2.load(model_name)

plot_results(log_dir)

# Enjoy trained agent
watch_agent = input("Do you want to watch your sick gaits? (Y/n):")
while watch_agent == "y" or "Y":
  obs = env.reset()
  for i in range(1000):
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render(mode='rgb_array')
  watch_agent = input("Do you want to watch your sick gaits? (Y/n):")
  if watch_agent == "n":
    break




# continue_training = input("Do you want to keen working on your sick gaits?(Y/n): "

# if continue_training == "Y":
#   print("Pussy, get out the gym")
# else:
#   extra_steps = input("Sweeeet, how many more timesteps?: ")
#   print (extra_steps + " more time steps it is..")

# watch for ever    
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()