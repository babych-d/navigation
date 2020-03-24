import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment
from argparse import ArgumentParser

from agent import Agent

SAVE_EVERY = 100
OUTPUT_FOLDER = "output"


def dqn(env, agent, brain_name,
        n_episodes=1000,
        max_t=10000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        cnn=False):
    """Deep Q-Learning.

    Params:
        env (UnityEnvironment): environment to train agent in
        agent (Agent): agent to train
        brain_name (str): name of the brain to use
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    best_score = 0
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment

        if cnn:
            state = env_info.visual_observations[0]
            n, x, y, c = state.shape
            state = np.reshape(state, (n, c, x, y))
        else:
            state = env_info.vector_observations[0]  # get the current state

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)

            env_info = env.step(action)[brain_name]  # send the action to the environment

            if cnn:
                state = env_info.visual_observations[0]
                n, x, y, c = state.shape
                next_state = np.reshape(state, (n, c, x, y))
            else:
                next_state = env_info.vector_observations[0]  # get the current state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        current_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, current_score), end="")
        if i_episode % SAVE_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, current_score))
            if current_score > best_score:
                output_file = os.path.join(OUTPUT_FOLDER, "best_model.pth")
                print(f"New high score, saving best model: {output_file}")
                torch.save(agent.qnetwork_local.state_dict(), output_file)
    return scores


def main(env_file, cnn=False):
    env = UnityEnvironment(file_name=env_file)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    if cnn:
        state = env_info.visual_observations[0]
    else:
        state = env_info.vector_observations[0]
    state_shape = state.shape
    print('States have length:', state_shape)

    agent = Agent(state_shape=state_shape, action_size=action_size, seed=0, cnn=cnn)

    print("Start training...")
    scores = dqn(env, agent, brain_name, cnn=cnn)
    fig = plt.figure()
    _ = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(os.path.join(OUTPUT_FOLDER, "train_scores.png"))


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--env_file", required=True)
    parser.add_argument("--cnn", action="store_true", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args.env_file, args.cnn)
