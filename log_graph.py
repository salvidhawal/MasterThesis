import json
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import math
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd

sns.set()
f = open("Logs//dqn_log.json", )

data_16 = json.load(f)

f.close()

f = open("Logs//dqn_log_600.json", )

data_600 = json.load(f)

f.close()

episode_reward_16 = data_16["episode_reward"]
episodes_16 = data_16["episode"]
episode_loss_16 = data_16["loss"]
episode_nb_steps_16 = data_16["nb_steps"]
episode_mean_eps_16 = data_16["mean_eps"]

episode_reward_600 = data_600["episode_reward"]

print(f"nb_steps: {episode_nb_steps_16[-1]}")
print(f"last 10 episode rewards: {episode_reward_16[-10:-1]}")

# LunarLander_with_labels_3500_good_reward - episode_reward

df_dqn_3500_good = pd.read_csv("Logs//dqn_3500_good.csv", delimiter=',', header=None)
episode_reward_3500_good = df_dqn_3500_good[1].values.tolist()
episode_reward_3500_good.pop(0)

df_dqn_1000_good = pd.read_csv("Logs//dqn_1000_good.csv", delimiter=',', header=None)
episode_reward_1000_good = df_dqn_1000_good[1].values.tolist()
episode_reward_1000_good.pop(0)

for i in range(len(episode_reward_3500_good)):
    episode_reward_3500_good[i] = float(episode_reward_3500_good[i])

for i in range(len(episode_reward_1000_good)):
    episode_reward_1000_good[i] = float(episode_reward_1000_good[i])

i = 0
for el in episode_loss_16:
    if math.isnan(el):
        episode_loss_16[i] = 0
        episode_mean_eps_16[i] = 0
    i += 1

epoch_16 = []
epoch_600 = []
epoch_3500_good = []
epoch_1000_good = []
rewards_16 = []
rewards_600 = []
rewards_3500_good = []
rewards_1000_good = []
loss = []
mean_eps = []

a = 499

for i in range(len(episode_reward_3500_good) - a):
    rewards_3500_good.append(mean(episode_reward_3500_good[i:i + a]))
    epoch_3500_good.append(i)

for i in range(len(episode_reward_1000_good) - a):
    rewards_1000_good.append(mean(episode_reward_1000_good[i:i + a]))
    epoch_1000_good.append(i)

j = 0
for i in range(len(episode_reward_16) - a):
    rewards_16.append(mean(episode_reward_16[i:i + a]))
    # loss.append(mean(episode_loss_16[i:i + a]))
    # mean_eps.append(mean(episode_mean_eps_16[i:i + a]))
    epoch_16.append(i)

for i in range(len(episode_reward_600) - a):
    rewards_600.append(mean(episode_reward_600[i:i + a]))
    epoch_600.append(i)

print(len(rewards_16), len(epoch_16), len(epoch_16)+a)
print(len(rewards_600), len(epoch_600))
print(len(rewards_3500_good), len(epoch_3500_good))

xnew_16 = np.linspace(min(epoch_16), max(epoch_16), 200)
xnew_600 = np.linspace(min(epoch_600), max(epoch_600), 200)
xnew_3500_good = np.linspace(min(epoch_3500_good), max(epoch_3500_good), 200)
xnew_1000_good = np.linspace(min(epoch_1000_good), max(epoch_1000_good), 200)

# define spline
spl_16 = make_interp_spline(epoch_16, rewards_16, k=3)
y_smooth_16 = spl_16(xnew_16)

spl_600 = make_interp_spline(epoch_600, rewards_600, k=3)
y_smooth_600 = spl_600(xnew_600)

spl_3500 = make_interp_spline(epoch_3500_good, rewards_3500_good, k=3)
y_smooth_3500 = spl_3500(xnew_3500_good)

spl_loss_1000 = make_interp_spline(epoch_1000_good, rewards_1000_good, k=3)
y_smooth_loss_1000 = spl_loss_1000(xnew_1000_good)

for i in range(1, 5):
    plt.subplot(2, 2, i)

    if i == 1:
        sns.lineplot(x=xnew_16, y=y_smooth_16, legend="auto", color="red")
        sns.lineplot(x=xnew_600, y=y_smooth_600, legend="auto", color="grey")
        sns.lineplot(x=xnew_1000_good, y=y_smooth_loss_1000, legend="auto", color="grey")
        sns.lineplot(x=xnew_3500_good, y=y_smooth_3500, legend="auto", color="grey")

        plt.legend(labels=["16 ins", "600 ins", "refrence - 3500 ins", "refrence - 1000 ins"])
        plt.xlabel("iteration")
        plt.ylabel("performance")

    elif i == 2:
        sns.lineplot(x=xnew_16, y=y_smooth_16, legend="auto", color="grey")
        sns.lineplot(x=xnew_600, y=y_smooth_600, legend="auto", color="blue")
        sns.lineplot(x=xnew_1000_good, y=y_smooth_loss_1000, legend="auto", color="grey")
        sns.lineplot(x=xnew_3500_good, y=y_smooth_3500, legend="auto", color="grey")

        plt.legend(labels=["16 ins", "600 ins", "refrence - 3500 ins", "refrence - 1000 ins"])
        plt.xlabel("iteration")
        plt.ylabel("performance")

    elif i == 3:
        sns.lineplot(x=xnew_16, y=y_smooth_16, legend="auto", color="grey")
        sns.lineplot(x=xnew_600, y=y_smooth_600, legend="auto", color="grey")
        sns.lineplot(x=xnew_1000_good, y=y_smooth_loss_1000, legend="auto", color="green")
        sns.lineplot(x=xnew_3500_good, y=y_smooth_3500, legend="auto", color="grey")

        plt.legend(labels=["16 ins", "600 ins", "refrence - 3500 ins", "refrence - 1000 ins"])
        plt.xlabel("iteration")
        plt.ylabel("performance")

    elif i == 4:
        sns.lineplot(x=xnew_16, y=y_smooth_16, legend="auto", color="grey")
        sns.lineplot(x=xnew_600, y=y_smooth_600, legend="auto", color="grey")
        sns.lineplot(x=xnew_1000_good, y=y_smooth_loss_1000, legend="auto", color="grey")
        sns.lineplot(x=xnew_3500_good, y=y_smooth_3500, legend="auto", color="orange")

        plt.legend(labels=["16 ins", "600 ins", "refrence - 3500 ins", "refrence - 1000 ins"])
        plt.xlabel("iteration")
        plt.ylabel("performance")

plt.show()
