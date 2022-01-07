import json
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import math
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd

sns.set()


def open_file(name):
    f = open(name, )
    data = json.load(f)
    f.close()
    return data


def return_avrage_rewards(episode_reward, avg):
    reward = []
    epoch = []
    for i in range(len(episode_reward) - avg):
        reward.append(mean(episode_reward[i:i + avg]))
        epoch.append(i)

    return reward, epoch


def return_xnew_y_smooth(epoch, reward):
    xnew = np.linspace(min(epoch), max(epoch), 200)
    spl = make_interp_spline(epoch, reward, k=3)
    y_smooth = spl(xnew)

    return xnew, y_smooth


no_ins = 4

#data_transformer_true = open_file(name=f"Logs//{no_ins}//dqn_log_{no_ins}_transformers_True.json")
#data_lstm_true = open_file(name=f"Logs//{no_ins}//dqn_log_{no_ins}_lstm_True.json")
#data_control_vector_true = open_file(name=f"Logs//{no_ins}//dqn_log_{no_ins}_control_vector_True.json")
data_transformer_false = open_file(name=f"Logs//{no_ins}//dqn_log_{no_ins}_transformers_False.json")
data_lstm_false = open_file(name=f"Logs//{no_ins}//dqn_log_{no_ins}_lstm_False.json")
data_control_vector_false = open_file(name=f"Logs//{no_ins}//dqn_log_{no_ins}_control_vector_False.json")


#episode_reward_transformer_true = data_transformer_true["episode_reward"]
#episode_reward_lstm_true = data_lstm_true["episode_reward"]
#episode_reward_control_vector_true = data_control_vector_true["episode_reward"]
episode_reward_transformer_false = data_transformer_false["episode_reward"]
episode_reward_lstm_false = data_lstm_false["episode_reward"]
episode_reward_control_vector_false = data_control_vector_false["episode_reward"]

avg = 299

#avg_reward_transformer_true, epoch_transformer_true = return_avrage_rewards(episode_reward_transformer_true, avg)
#avg_reward_lstm_true, epoch_lstm_true = return_avrage_rewards(episode_reward_lstm_true, avg)
#avg_reward_control_vector_true, epoch_control_vector_true = return_avrage_rewards(episode_reward_control_vector_true, avg)
avg_reward_transformer_false, epoch_transformer_false = return_avrage_rewards(episode_reward_transformer_false, avg)
avg_reward_lstm_false, epoch_lstm_false = return_avrage_rewards(episode_reward_lstm_false, avg)
avg_reward_control_vector_false, epoch_control_vector_false = return_avrage_rewards(episode_reward_control_vector_false, avg)

#xnew_transformer_true, y_smooth_transformer_true = return_xnew_y_smooth(epoch_transformer_true, avg_reward_transformer_true)
#xnew_lstm_true, y_smooth_lstm_true = return_xnew_y_smooth(epoch_lstm_true, avg_reward_lstm_true)
#xnew_control_vector_true, y_smooth_control_vector_true = return_xnew_y_smooth(epoch_control_vector_true, avg_reward_control_vector_true)
xnew_transformer_false, y_smooth_transformer_false = return_xnew_y_smooth(epoch_transformer_false, avg_reward_transformer_false)
xnew_lstm_false, y_smooth_lstm_false = return_xnew_y_smooth(epoch_lstm_false, avg_reward_lstm_false)
xnew_control_vector_false, y_smooth_control_vector_false = return_xnew_y_smooth(epoch_control_vector_false, avg_reward_control_vector_false)

for i in range(1, 5):
    plt.subplot(2, 2, i)

    if i == 1:
        sns.lineplot(x=xnew_transformer_true, y=y_smooth_transformer_true, legend="auto", color="red")
        sns.lineplot(x=xnew_lstm_true, y=y_smooth_lstm_true, legend="auto", color="grey")
        sns.lineplot(x=xnew_control_vector_true, y=y_smooth_control_vector_true, legend="auto", color="grey")
        sns.lineplot(x=xnew_transformer_false, y=y_smooth_transformer_false, legend="auto", color="red", linestyle="--")
        sns.lineplot(x=xnew_lstm_false, y=y_smooth_lstm_false, legend="auto", color="grey", linestyle="--")
        sns.lineplot(x=xnew_control_vector_false, y=y_smooth_control_vector_false, legend="auto", color="grey", linestyle="--")

        plt.legend(labels=["Transformer - CT", "Lstm - CT", "Control Vector - CT", "Transformer - CF", "Lstm - CF", "Control Vector - CF"])
        plt.xlabel("iteration")
        plt.ylabel("performance")

    elif i == 2:
        sns.lineplot(x=xnew_transformer_true, y=y_smooth_transformer_true, legend="auto", color="grey")
        sns.lineplot(x=xnew_lstm_true, y=y_smooth_lstm_true, legend="auto", color="blue")
        sns.lineplot(x=xnew_control_vector_true, y=y_smooth_control_vector_true, legend="auto", color="grey")
        sns.lineplot(x=xnew_transformer_false, y=y_smooth_transformer_false, legend="auto", color="grey", linestyle="--")
        sns.lineplot(x=xnew_lstm_false, y=y_smooth_lstm_false, legend="auto", color="blue", linestyle="--")
        sns.lineplot(x=xnew_control_vector_false, y=y_smooth_control_vector_false, legend="auto", color="grey", linestyle="--")

        plt.legend(labels=["Transformer - CT", "Lstm - CT", "Control Vector - CT", "Transformer - CF", "Lstm - CF", "Control Vector - CF"])
        plt.xlabel("iteration")
        plt.ylabel("performance")

    elif i == 3:
        sns.lineplot(x=xnew_transformer_true, y=y_smooth_transformer_true, legend="auto", color="grey")
        sns.lineplot(x=xnew_lstm_true, y=y_smooth_lstm_true, legend="auto", color="grey")
        sns.lineplot(x=xnew_control_vector_true, y=y_smooth_control_vector_true, legend="auto", color="green")
        sns.lineplot(x=xnew_transformer_false, y=y_smooth_transformer_false, legend="auto", color="grey", linestyle="--")
        sns.lineplot(x=xnew_lstm_false, y=y_smooth_lstm_false, legend="auto", color="grey", linestyle="--")
        sns.lineplot(x=xnew_control_vector_false, y=y_smooth_control_vector_false, legend="auto", color="green", linestyle="--")

        plt.legend(labels=["Transformer - CT", "Lstm - CT", "Control Vector - CT", "Transformer - CF", "Lstm - CF", "Control Vector - CF"])
        plt.xlabel("iteration")
        plt.ylabel("performance")

    elif i == 4:
        sns.lineplot(x=xnew_transformer_true, y=y_smooth_transformer_true, legend="auto", color="red")
        sns.lineplot(x=xnew_lstm_true, y=y_smooth_lstm_true, legend="auto", color="blue")
        sns.lineplot(x=xnew_control_vector_true, y=y_smooth_control_vector_true, legend="auto", color="green")
        sns.lineplot(x=xnew_transformer_false, y=y_smooth_transformer_false, legend="auto", color="red", linestyle="--")
        sns.lineplot(x=xnew_lstm_false, y=y_smooth_lstm_false, legend="auto", color="blue", linestyle="--")
        sns.lineplot(x=xnew_control_vector_false, y=y_smooth_control_vector_false, legend="auto", color="green", linestyle="--")

        plt.legend(labels=["Transformer - CT", "Lstm - CT", "Control Vector - CT", "Transformer - CF", "Lstm - CF", "Control Vector - CF"])
        plt.xlabel("iteration")
        plt.ylabel("performance")

plt.show()
