import os
import argparse
import time

from envs.atari_env import AtariEnv
from ppo_agent import PPOAgent, PPOConfig


def make_env(render_mode='human'):
    return AtariEnv(env_name='ALE/AirRaid-v5', frame_skip=4, frame_stack=4, screen_size=84, render_mode=render_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render_mode', type=str, default='human')
    parser.add_argument('--sleep_time', type=float, default=0.0, help='Seconds to sleep per step when rendering')
    args = parser.parse_args()

    env = make_env(render_mode=args.render_mode)
    num_actions = env.action_space.n
    in_channels = env.observation_space.shape[2]

    agent = PPOAgent(in_channels=in_channels, num_actions=num_actions, config=PPOConfig())
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    episode_loaded = agent.load(args.model_path)
    print(f"Loaded checkpoint from episode {episode_loaded} -> {args.model_path}")

    for ep in range(1, args.num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _, _ = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if args.render_mode == 'human' and args.sleep_time > 0:
                time.sleep(args.sleep_time)
        print(f"Eval Episode {ep}: reward={ep_reward:.2f}")

    env.close()


if __name__ == '__main__':
    main()
