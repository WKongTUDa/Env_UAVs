from grid_world_env import ParallelGridGame

import random

def random_demo(env, num_cycles=100, render=False):
    observations, _ = env.reset()

    env.render(save_dir='figure')

    for _ in range(num_cycles):
        # actions = [env.action_space(agent).sample() for agent in env.agents if agent != "null"]
        actions = [env.action_space(agent).sample() if agent != "null" else -1 for agent in env.agents]
        observations, reward, done, _, _, _ = env.step(actions)

        if render:
            env.render(save_dir='figure')

        if done:
            print(reward)
            break

if __name__ == "__main__":
    env = ParallelGridGame()
    random_demo(env, num_cycles=100, render=True)

