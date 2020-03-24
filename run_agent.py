from unityagents import UnityEnvironment
from argparse import ArgumentParser
import torch

from agent import Agent


def run(env_file, model_file, num_episodes=5):
    env = UnityEnvironment(file_name=env_file)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    state_shape = state.shape
    agent = Agent(state_shape=state_shape, action_size=action_size, seed=0)

    agent.qnetwork_local.load_state_dict(torch.load(model_file))

    for i in range(num_episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

        print("Score: {}".format(score))
    env.close()


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--env_file", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--num_episodes", default=3, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.env_file, args.model_file, num_episodes=args.num_episodes)
