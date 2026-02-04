import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from envs.atari_env import AtariEnv


def make_env(render_mode='rgb_array'):
    # 训练期间关闭死亡惩罚，避免早期学习被-50主导
    return AtariEnv(env_name='ALE/AirRaid-v5', frame_skip=4, frame_stack=4, screen_size=84, render_mode=render_mode, death_penalty=0)


class PolicyCNN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.net(dummy)
        conv_flat = conv_out.view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def preprocess_obs(obs_np):
    obs_chw = np.transpose(obs_np, (2, 0, 1))
    obs_t = torch.from_numpy(obs_chw).float() / 255.0
    return obs_t


def grpo_update(policy, optimizer, observations, actions, returns, entropy_coef=0.01):
    logits = policy(observations)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    chosen_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    baseline = returns.mean().detach()
    advantages = returns - baseline
    loss = -(chosen_log_probs * advantages).mean() - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
    return loss.item(), entropy.item(), advantages.mean().item()


def collect_rollout(env, policy, device, rollout_steps, gamma=0.99):
    obs_list = []
    act_list = []
    rew_list = []
    done_list = []
    episodes_completed = 0

    obs_np, _ = env.reset()
    transitions = 0

    for _ in range(rollout_steps):
        obs_t = preprocess_obs(obs_np).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs.squeeze(0), num_samples=1).item()
        next_obs_np, reward, terminated, truncated, _ = env.step(action)

        obs_list.append(obs_np)
        act_list.append(action)
        rew_list.append(reward)
        done_list.append(terminated or truncated)
        transitions += 1

        obs_np = next_obs_np
        if terminated or truncated:
            episodes_completed += 1
            obs_np, _ = env.reset()

    returns = []
    G = 0.0
    for r, d in zip(reversed(rew_list), reversed(done_list)):
        if d:
            G = 0.0
        G = r + gamma * G
        returns.append(G)
    returns.reverse()

    obs_batch = torch.stack([preprocess_obs(o) for o in obs_list], dim=0).to(device)
    act_batch = torch.tensor(act_list, dtype=torch.long, device=device)
    ret_batch = torch.tensor(returns, dtype=torch.float32, device=device)

    return obs_batch, act_batch, ret_batch, transitions, episodes_completed


def plot_avg_rewards(avg_history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    episodes = [ep for ep, _ in avg_history]
    avgs = [avg for _, avg in avg_history]
    plt.plot(episodes, avgs, label='Average Reward (last 100)')
    try:
        marker_interval = 10
        marker_eps = [ep for ep in episodes if ep % marker_interval == 0]
        marker_vals = [avg for ep, avg in avg_history if ep % marker_interval == 0]
        if marker_eps:
            plt.scatter(marker_eps, marker_vals, color='red', s=30, zorder=3)
    except Exception:
        pass
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_episode_rewards(step_rewards, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    steps = list(range(1, len(step_rewards) + 1))
    cum = np.cumsum(step_rewards)
    plt.plot(steps, step_rewards, label='Step Reward', alpha=0.5)
    plt.plot(steps, cum, label='Cumulative Reward', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Episode Reward Trajectory')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=200000, help='Total env steps to train')
    parser.add_argument('--rollout_steps', type=int, default=2048, help='Steps per GRPO update')
    parser.add_argument('--checkpoint_every', type=int, default=100, help='Save checkpoint frequency (episodes or steps based)')
    parser.add_argument('--save_on', type=str, choices=['episodes','steps'], default='episodes', help='Trigger checkpoint by episodes or steps')
    parser.add_argument('--target_episodes', type=int, default=None, help='Stop training after N completed episodes (overrides num_steps if set)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy_coef', type=float, default=0.05)
    args = parser.parse_args()

    env = make_env(render_mode='rgb_array')
    num_actions = env.action_space.n
    in_channels = env.observation_space.shape[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = PolicyCNN(in_channels=in_channels, num_actions=num_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    step_count = 0
    episode_count = 0
    reward_hist = deque(maxlen=100)
    avg_history = []
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    avg_txt = os.path.join('logs', 'avg_reward.txt')

    # Resume if provided
    if args.resume_path and os.path.isfile(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        state = ckpt.get('model_state', ckpt)
        policy.load_state_dict(state)
        step_count = ckpt.get('step_count', 0)
        episode_count = ckpt.get('episode_count', 0)
        print(f"Resumed from {args.resume_path}: steps={step_count}, episodes={episode_count}")

    while True:
        obs_batch, act_batch, ret_batch, transitions, episodes_completed = collect_rollout(
            env, policy, device, args.rollout_steps, args.gamma
        )
        loss, entropy, adv_mean = grpo_update(policy, optimizer, obs_batch, act_batch, ret_batch, args.entropy_coef)
        prev_step_count = step_count
        step_count += transitions
        prev_episode_count = episode_count
        episode_count += episodes_completed

        # Diagnostics episode to update avg reward plot
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step_rewards = []
        while not done:
            obs_t = preprocess_obs(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = policy(obs_t)
                probs = torch.softmax(logits, dim=-1)
                # 诊断回合使用采样而非贪心，更能反映探索下的平均表现
                action = torch.multinomial(probs.squeeze(0), num_samples=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step_rewards.append(reward)
        reward_hist.append(ep_reward)
        avg_reward = float(np.mean(reward_hist))
        avg_history.append((episode_count, avg_reward))
        with open(avg_txt, 'a', encoding='utf-8') as f:
            f.write(f"{episode_count},{avg_reward}\n")
        # 覆盖式绘图：仅保存平均回报曲线到同一文件
        plot_avg_rewards(avg_history, os.path.join('plots', 'avg_reward.png'))
        print(f"[EP] {episode_count} return={ep_reward:.2f} avg(100)={avg_reward:.2f} plot=plots/avg_reward.png")

        # Save checkpoint based on configured trigger
        if args.save_on == 'episodes':
            if episode_count // args.checkpoint_every != prev_episode_count // args.checkpoint_every and episodes_completed > 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"grpo_checkpoint_episode_{episode_count}.pth")
                torch.save({
                    'model_state': policy.state_dict(),
                    'step_count': step_count,
                    'episode_count': episode_count,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
        else:  # steps
            if step_count // args.checkpoint_every != prev_step_count // args.checkpoint_every:
                ckpt_path = os.path.join(args.checkpoint_dir, f"grpo_checkpoint_step_{step_count:07d}.pth")
                torch.save({
                    'model_state': policy.state_dict(),
                    'step_count': step_count,
                    'episode_count': episode_count,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # Stopping condition: target episodes reached, or fallback to num_steps
        if args.target_episodes is not None and episode_count >= args.target_episodes:
            break
        if args.target_episodes is None and step_count >= args.num_steps:
            break

    final_ckpt = os.path.join(args.checkpoint_dir, f"grpo_checkpoint_episode_{episode_count}.pth")
    torch.save({
        'model_state': policy.state_dict(),
        'step_count': step_count,
        'episode_count': episode_count,
    }, final_ckpt)
    print(f"Training complete. Final checkpoint: {final_ckpt}")


if __name__ == '__main__':
    main()
