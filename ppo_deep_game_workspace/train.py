import os
import argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from envs.atari_env import AtariEnv
from ppo_agent import PPOAgent, PPOConfig


def make_env(render_mode='rgb_array', screen_size: int = 84):
    return AtariEnv(env_name='ALE/AirRaid-v5', frame_skip=4, frame_stack=4, screen_size=screen_size, render_mode=render_mode)


def plot_avg_rewards(avg_history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    episodes = [ep for ep, _ in avg_history]
    avgs = [avg for _, avg in avg_history]
    plt.plot(episodes, avgs, label='Average Reward (last 100)')
    # Mark and annotate every 10 episodes by default
    try:
        marker_interval = 10
        marker_eps = [ep for ep in episodes if ep % marker_interval == 0]
        marker_vals = [avg for ep, avg in avg_history if ep % marker_interval == 0]
        if marker_eps:
            plt.scatter(marker_eps, marker_vals, color='red', s=30, zorder=3)
            for mep, mval in zip(marker_eps, marker_vals):
                plt.annotate(f"{mval:.1f}", (mep, mval), textcoords="offset points", xytext=(0,6), ha='center', fontsize=8)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--update_steps', type=int, default=4096, help='Steps per PPO update')
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size for PPO updates')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=3e-6)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--screen_size', type=int, default=84, help='Observation height/width (e.g., 128 for larger input)')
    args = parser.parse_args()

    env = make_env(render_mode='rgb_array', screen_size=args.screen_size)
    num_actions = env.action_space.n
    in_channels = env.observation_space.shape[2]
    input_size = env.observation_space.shape[0]

    cfg = PPOConfig(learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda, batch_size=args.batch_size)
    agent = PPOAgent(in_channels=in_channels, num_actions=num_actions, config=cfg, input_size=input_size)

    start_episode = 1
    if args.resume_path:
        if os.path.exists(args.resume_path):
            start_episode = agent.load(args.resume_path) + 1
            print(f"Resumed from {args.resume_path}, starting at episode {start_episode}")
        else:
            print(f"Resume path {args.resume_path} not found; starting fresh.")

    reward_hist = deque(maxlen=100)
    avg_history = []  # list of (episode, avg)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    avg_txt = os.path.join('logs', 'avg_reward.txt')

    # Trajectory buffer
    trajectories = []
    steps_since_update = 0

    for episode in range(start_episode, args.num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        traj = {'obs': [], 'actions': [], 'logprobs': [], 'values': [], 'rewards': [], 'dones': [], 'next_value': 0.0}

        while not done:
            action, logprob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            traj['obs'].append(obs)
            traj['actions'].append(action)
            traj['logprobs'].append(logprob)
            traj['values'].append(value)
            traj['rewards'].append(reward)
            traj['dones'].append(1.0 if done else 0.0)

            obs = next_obs
            ep_reward += reward
            steps_since_update += 1

            if steps_since_update >= args.update_steps:
                # Bootstrap next value for last transition
                _, _, next_val = agent.act(obs)
                traj['next_value'] = next_val
                trajectories.append(traj)
                # Debug info before update
                total_traj_steps = sum(len(t['obs']) for t in trajectories)
                print(f"[DEBUG] Triggering update: steps_since_update={steps_since_update}, total_traj_steps={total_traj_steps}, n_trajectories={len(trajectories)}, update_steps={args.update_steps}, batch_size={args.batch_size}")
                stats = agent.update(trajectories)
                trajectories.clear()
                steps_since_update = 0
                print(f"Update: steps={stats['n_steps']}, policy_loss={stats['policy_loss']:.4f}, value_loss={stats['value_loss']:.4f}, entropy={stats['entropy']:.4f}")
                # IMPORTANT: start a new trajectory segment after an update to avoid reusing
                # previously consumed transitions and ensure correct bootstrapping.
                traj = {'obs': [], 'actions': [], 'logprobs': [], 'values': [], 'rewards': [], 'dones': [], 'next_value': 0.0}

        # Episode finished
        print(f"Episode {episode}: reward={ep_reward:.2f}")
        reward_hist.append(ep_reward)
        avg_reward = float(np.mean(reward_hist))
        avg_history.append((episode, avg_reward))
        with open(avg_txt, 'a', encoding='utf-8') as f:
            f.write(f"{episode},{avg_reward}\n")
        # Only update the saved plot every 10 episodes to reduce IO and show interval markers
        if episode % 10 == 0:
            plot_avg_rewards(avg_history, os.path.join('plots', 'avg_reward.png'))

        # Finish trajectory and maybe update
        _, _, next_val = agent.act(obs)
        traj['next_value'] = next_val
        trajectories.append(traj)

        # Optional update if we have enough data (e.g., across episodes)
        total_steps = sum(len(t['obs']) for t in trajectories)
        if total_steps >= args.update_steps:
            print(f"[DEBUG] Post-episode update trigger: total_steps={total_steps}, n_trajectories={len(trajectories)}, update_steps={args.update_steps}, batch_size={args.batch_size}")
            stats = agent.update(trajectories)
            trajectories.clear()
            steps_since_update = 0
            print(f"Post-episode update: steps={stats['n_steps']} policy_loss={stats['policy_loss']:.4f}")

        # Save checkpoint every N episodes
        if episode % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ppo_checkpoint_episode_{episode}.pth")
            agent.save(ckpt_path, episode)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_ckpt = os.path.join(args.checkpoint_dir, f"ppo_checkpoint_episode_{args.num_episodes}.pth")
    agent.save(final_ckpt, args.num_episodes)
    print(f"Training complete. Final checkpoint: {final_ckpt}")


if __name__ == '__main__':
    main()
