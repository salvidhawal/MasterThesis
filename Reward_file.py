import numpy as np
import time


def side_power_reward(label, reward, s_power, r_power, l_power, x):
    if label == 0:
        if -0.15 <= x <= 0.15:
            reward -= s_power * 0.03
        else:
            if 0.15 <= x <= 1.0:
                reward += r_power * 0.03
                reward -= l_power * 0.03
            elif -1.0 <= x <= -0.15:
                reward -= r_power * 0.03
                reward += l_power * 0.03

    elif label == 1:
        if 0.15 <= x <= 0.25:
            reward -= s_power * 0.03
        else:
            if 0.25 <= x <= 1.0:
                reward += r_power * 0.03
                reward -= l_power * 0.03
            elif -1.0 <= x <= 0.15:
                reward -= r_power * 0.03
                reward += l_power * 0.03
    elif label == 2:
        if -0.25 <= x <= -0.15:
            reward -= s_power * 0.03
        else:
            if -0.15 <= x <= 1.0:
                reward += r_power * 0.03
                reward -= l_power * 0.03
            elif -1.0 <= x <= -0.25:
                reward -= r_power * 0.03
                reward += l_power * 0.03
    elif label == 3:
        if -1 <= x <= -0.25 or 0.25 <= x <= 1:
            reward -= s_power * 0.03
        else:
            if -0.25 <= x <= 0.0:
                reward += r_power * 0.03
                reward -= l_power * 0.03
            elif 0.0 <= x <= 0.25:
                reward -= r_power * 0.03
                reward += l_power * 0.03
    return reward


def game_over_reward(label, reward, x, at_fail):
    if label == 0:
        if -0.15 <= x <= 0.15:
            reward += 25
        else:
            reward -= at_fail
    elif label == 1:
        if 0.15 <= x <= 0.25:
            reward += 25
        else:
            reward -= at_fail
    elif label == 2:
        if -0.25 <= x <= -0.15:
            reward += 25
        else:
            reward -= at_fail
    elif label == 3:
        if -1 <= x <= -0.25 or 0.25 <= x <= 1:
            reward += 25
        else:
            reward -= at_fail

    return reward


def touch_leg_reward(label, shaping, x, touch_points, no_touch_points, leg_1, leg_2):
    if label == 0:
        if -0.15 <= x <= 0.15:
            shaping = shaping + (touch_points * leg_1) + (
                    touch_points * leg_2)  # And ten points for legs contact, the idea is if you
            # lose contact again after landing, you get negative reward
        else:
            shaping = shaping + (no_touch_points * leg_1) + (
                    no_touch_points * leg_2)  # And ten points for legs contact, the idea is if you
            # lose contact again after landing, you get negative reward

    elif label == 1:
        if 0.15 <= x <= 0.25:
            shaping = shaping + (touch_points * leg_1) + (
                    touch_points * leg_2)  # And ten points for legs contact, the idea is if you
            # lose contact again after landing, you get negative reward
        else:
            shaping = shaping + (no_touch_points * leg_1) + (no_touch_points * leg_2)

    elif label == 2:
        if -0.25 <= x <= -0.15:
            shaping = shaping + (touch_points * leg_1) + (
                    touch_points * leg_2)  # And ten points for legs contact, the idea is if you
            # lose contact again after landing, you get negative reward
        else:
            shaping = shaping + (no_touch_points * leg_1) + (no_touch_points * leg_2)

    elif label == 3:
        if -1 <= x <= -0.25 or 0.25 <= x <= 1:
            shaping = shaping + (touch_points * leg_1) + (
                    touch_points * leg_2)  # And ten points for legs contact, the idea is if you
            # lose contact again after landing, you get negative reward
        else:
            shaping = shaping + (no_touch_points * leg_1) + (no_touch_points * leg_2)

    return shaping


def reward_shaping(action, label, state, prev_shaping, lander_awake, game_over, phase, dropdown):
    global y
    m_power = 0.0
    s_power = 0.0
    l_power = 0.0
    r_power = 0.0

    if action == 2:
        m_power = 1.0
    elif action == 3:
        s_power = 1.0
        r_power = 1.0
    elif action == 1:
        l_power = 1.0
        s_power = 1.0

    reward = 0
    shaping = 0

    x = state[0]
    if phase == 1:
        y = state[1]
    elif phase == 2:
        if dropdown:
            y = state[1]
        elif not dropdown:
            y = 1.4 - state[1]

    # print(f"x: {x}, y: {y}")
    if label == 0:
        shaping = -100 * np.sqrt((x * x) + (y * y))
    elif label == 1:
        if -1 <= x <= 0.2:
            shaping = -100 * np.sqrt((x - 0.2) * (x - 0.2) + (y * y))
        else:
            shaping = -150 * np.sqrt((x - 0.2) * (x - 0.2) + ((y / 1.5) * (y / 1.5)))
    elif label == 2:
        if -0.2 <= x <= 1.01:
            shaping = -100 * np.sqrt((x + 0.2) * (x + 0.2) + (y * y))
        else:
            shaping = -150 * np.sqrt((x + 0.2) * (x + 0.2) + ((y / 1.5) * (y / 1.5)))
    elif label == 3:
        if -1 <= x <= 0.0:
            if -1 <= x <= -0.625:
                shaping = -180 * np.sqrt((x + 0.625) * (x + 0.625) + ((y / 1.5) * (y / 1.5)))
            else:
                shaping = -120 * np.sqrt((x + 0.625) * (x + 0.625) + (y * y))
        elif 0.0 <= x <= 1:
            if 0.625 <= x <= 1.01:
                shaping = -180 * np.sqrt((x - 0.625) * (x - 0.625) + ((y / 1.5) * (y / 1.5)))
            else:
                shaping = -120 * np.sqrt((x - 0.625) * (x - 0.625) + (y * y))
    # print(f"direction shaping 1: {shaping}")

    if phase == 1:
        shaping = shaping + (-100 * np.sqrt(state[2] * state[2] + state[3] * state[3])) + (-100 * abs(state[4]))
        shaping = touch_leg_reward(label=label, shaping=shaping, x=x, touch_points=10, no_touch_points=5,
                                   leg_1=state[6], leg_2=state[7])


    elif phase == 2:
        shaping = shaping + (-100 * np.sqrt(state[2] * state[2] + state[3] * state[3])) + (-100 * abs(state[4]))
        if dropdown:
            shaping = touch_leg_reward(label=label, shaping=shaping, x=x, touch_points=10, no_touch_points=5,
                                       leg_1=state[6], leg_2=state[7])
        elif not dropdown:
            shaping = touch_leg_reward(label=label, shaping=shaping, x=x, touch_points=2, no_touch_points=-2,
                                       leg_1=state[6], leg_2=state[7])

    # print(f"overall shaping 2: {shaping}")

    if prev_shaping is not None:
        reward = shaping - prev_shaping
    prev_shaping = shaping

    # print(f'reward and prev_shaping: {reward}, {prev_shaping}')

    if phase == 1:
        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heuristic landing
        # reward -= s_power * 0.03
        reward = side_power_reward(label=label, reward=reward, s_power=s_power, r_power=r_power, l_power=l_power, x=x)

    elif phase == 2:
        if dropdown:
            reward -= m_power * 0.30  # less fuel spent is better, about -30 for heuristic landing
            reward -= s_power * 0.03
            # reward = side_power_reward(label=label, reward=reward, s_power=s_power, r_power=r_power, l_power=l_power, x=x)

        elif not dropdown:
            reward += m_power * 0.80
            if action == 0:
                reward -= (1.0 * 0.50)
            reward -= s_power * 0.03
            # reward = side_power_reward(label=label, reward=reward, s_power=s_power, r_power=r_power, l_power=l_power, x=x)

    # print(f"phase: {phase}, engine reward 1 : {reward}")

    if game_over or abs(state[0]) >= 1.0:
        reward = -100
        # reward = game_over_reward(label=label, reward=reward, x=x, at_fail=0) # -75 -> -100
        # print(f"end game reward 2 : {reward}")
    if not lander_awake:
        # print(f"position of x: {x}, label: {label}")
        reward = 75
        reward = game_over_reward(label=label, reward=reward, x=x, at_fail=50)  # 25 -> 100

    return reward, prev_shaping
