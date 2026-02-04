import torch
import numpy as np
from envs.atari_env import AtariEnv
from models.dqn_model import DQN
import imageio
import matplotlib.pyplot as plt

def visualize_dqn(env_name='ALE/AirRaid-v5', model_path='checkpoints/dqn_episode_900.pth',
                  max_steps=1000, save_path='rollout.mp4', render_mode='human'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = AtariEnv(env_name, render_mode=render_mode)
    num_actions = env.action_space.n

    policy_net = DQN((4, 84, 84), num_actions).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it's a full checkpoint or just state_dict
    if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
        policy_net.load_state_dict(checkpoint['policy_net'])
    else:
        policy_net.load_state_dict(checkpoint)
    
    policy_net.eval()

    state, _ = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    frames = []
    rewards = []
    total_reward = 0

    for step in range(max_steps):
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = torch.from_numpy(next_state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        total_reward += reward
        rewards.append(total_reward)

        frame = env.render()
        if render_mode == 'rgb_array':
            frames.append(frame)

        if done:
            break

    env.close()

    if render_mode == 'rgb_array':
        # 保存视频
        imageio.mimsave(save_path, frames, fps=30)
        print(f"Video saved to {save_path}")

        # 绘制奖励曲线
        plt.plot(rewards)
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.title('DQN Performance on AirRaid-v5')
        plt.savefig('reward_plot.png')
        plt.show()

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize DQN performance on Atari games')
    parser.add_argument('--env_name', type=str, default='ALE/AirRaid-v5', help='Environment name')
    parser.add_argument('--model_path', type=str, default='checkpoints/dqn_checkpoint_episode_900.pth', help='Path to the model checkpoint')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps to visualize')
    parser.add_argument('--render_mode', type=str, default='human', help='Render mode: human for display, rgb_array for video')

    args = parser.parse_args()
    visualize_dqn(**vars(args))