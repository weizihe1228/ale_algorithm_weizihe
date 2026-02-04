import os
import argparse
import time
import torch
import numpy as np

from envs.atari_env import AtariEnv


def make_env(render_mode='human', frame_skip=4):
    # 评估时也可关闭死亡惩罚，观察基础 ALE 回报
    return AtariEnv(env_name='ALE/AirRaid-v5', frame_skip=frame_skip, frame_stack=4, screen_size=84, render_mode=render_mode, death_penalty=0)


class PolicyCNN(torch.nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.net(dummy)
        conv_flat = conv_out.view(1, -1).size(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_flat, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def preprocess_obs(obs_np):
    obs_chw = np.transpose(obs_np, (2, 0, 1))
    obs_t = torch.from_numpy(obs_chw).float() / 255.0
    return obs_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render_mode', type=str, default='human')
    parser.add_argument('--sleep_time', type=float, default=0.0, help='Seconds to sleep per step when rendering')
    parser.add_argument('--frame_skip', type=int, default=4, help='Env frame_skip for eval responsiveness')
    parser.add_argument('--sample_policy', action='store_true', help='Sample actions from policy instead of greedy')
    # Default to training-style sampling (softmax multinomial) with no epsilon
    parser.set_defaults(sample_policy=True)
    parser.add_argument('--epsilon', type=float, default=0.0, help='Epsilon-greedy random action probability (default 0.0 to match train)')
    args = parser.parse_args()

    env = make_env(render_mode=args.render_mode, frame_skip=args.frame_skip)
    num_actions = env.action_space.n
    in_channels = env.observation_space.shape[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = PolicyCNN(in_channels=in_channels, num_actions=num_actions).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get('model_state', ckpt)
    policy.load_state_dict(state)
    policy.eval()

    # Print action meanings for diagnostics
    try:
        meanings = env.env.unwrapped.get_action_meanings()
        print(f"Action meanings: {meanings}")
    except Exception:
        meanings = [str(i) for i in range(num_actions)]
        print(f"Action meanings unavailable; using indices 0..{num_actions-1}")

    for ep in range(1, args.num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        action_counts = np.zeros(num_actions, dtype=int)
        while not done:
            obs_t = preprocess_obs(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = policy(obs_t)
                if args.sample_policy:
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs.squeeze(0), num_samples=1).item()
                else:
                    action = torch.argmax(logits, dim=-1).item()
            # epsilon-greedy randomization (only if epsilon > 0)
            if args.epsilon > 0.0 and np.random.rand() < args.epsilon:
                action = int(np.random.randint(0, num_actions))
            action_counts[action] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if args.render_mode == 'human' and args.sleep_time > 0:
                time.sleep(args.sleep_time)
        # Print action distribution for this episode
        dist_str = ", ".join([f"{meanings[i]}:{action_counts[i]}" for i in range(num_actions)])
        print(f"Eval Episode {ep}: reward={ep_reward:.2f} | actions: {dist_str}")

    env.close()


if __name__ == '__main__':
    main()
