import argparse
import time
import numpy as np
import torch

from envs.atari_env import AtariEnv
from ppo_agent import PPOAgent, PPOConfig


def make_env(render_mode='human', screen_size: int = 84):
    return AtariEnv(env_name='ALE/AirRaid-v5', frame_skip=4, frame_stack=4, screen_size=screen_size, render_mode=render_mode)


def run_eval(model_path: str, num_episodes: int = 5, render_mode: str = 'human', sleep_time: float = 0.0, screen_size: int = 84):
    env = make_env(render_mode=render_mode, screen_size=screen_size)
    num_actions = env.action_space.n
    in_channels = env.observation_space.shape[2]
    input_size = env.observation_space.shape[0]

    agent = PPOAgent(in_channels=in_channels, num_actions=num_actions, config=PPOConfig(), input_size=input_size)
    if not torch.cuda.is_available():
        agent.config.device = 'cpu'
    agent.device = torch.device(agent.config.device)
    agent.policy.to(agent.device)

    if not model_path:
        raise ValueError('model_path is required')
    ckpt = torch.load(model_path, map_location=agent.device)
    agent.policy.load_state_dict(ckpt['model_state'])
    print(f"Loaded deterministic checkpoint from episode {ckpt.get('episode')} -> {model_path}")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # obs is HWC uint8 with stacked frames in last axis
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0
            obs_t = torch.from_numpy(obs_chw).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                logits, _ = agent.policy(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if render_mode == 'human' and sleep_time > 0:
                time.sleep(sleep_time)

        print(f"Deterministic Eval Episode {ep}: reward={ep_reward:.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--render_mode', type=str, default='human')
    parser.add_argument('--sleep_time', type=float, default=0.0)
    parser.add_argument('--screen_size', type=int, default=84)
    args = parser.parse_args()
    run_eval(args.model_path, args.num_episodes, args.render_mode, args.sleep_time, args.screen_size)


if __name__ == '__main__':
    main()
