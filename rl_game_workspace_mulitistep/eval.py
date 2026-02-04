import torch
import numpy as np
from envs.atari_env import AtariEnv
from models.dqn_model import DQN
import imageio
import os

def evaluate_dqn(env_name='ALE/AirRaid-v5', model_path='checkpoints/dqn_checkpoint_final.pth',
                 num_episodes=5, render=False, render_mode='rgb_array', save_video=False, video_path='evaluation.mp4',
                 print_detections=False, clear_screen=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = AtariEnv(env_name, render_mode=render_mode)
    num_actions = env.action_space.n

    policy_net = DQN((4, 84, 84), num_actions).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it's a full checkpoint or just state_dict
    if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
        # Full checkpoint
        policy_net.load_state_dict(checkpoint['policy_net'])
        print(f"Loaded full checkpoint from episode {checkpoint.get('episode', 'unknown')}")
    else:
        # Old checkpoint (only state_dict)
        policy_net.load_state_dict(checkpoint)
        print("Loaded state_dict checkpoint")
    
    policy_net.eval()

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        total_reward = 0
        done = False
        frames = []

        while not done:
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = torch.from_numpy(next_state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            total_reward += reward

            # Print bullet and enemy detections only if detected and enabled
            bullets = env._get_bullets(next_state)
            enemies = env._get_enemy_planes(next_state)
            if print_detections and (bullets or enemies):
                print(f"Step: Bullets detected: {len(bullets)} at {bullets}, Enemies detected: {len(enemies)} at {enemies}")
            elif clear_screen and not (bullets or enemies):
                os.system('cls')  # Clear screen when no detections and enabled

            if render_mode == 'human':
                env.render()
            elif render and render_mode == 'rgb_array':
                frame = env.render()
                frames.append(frame)

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

    if render_mode == 'rgb_array' and frames:
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")

    env.close()
    return avg_reward

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate DQN on Atari games')
    parser.add_argument('--env_name', type=str, default='ALE/AirRaid-v5', help='Environment name')
    parser.add_argument('--model_path', type=str, default='checkpoints/dqn_checkpoint_final.pth', help='Path to the model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--render_mode', type=str, default='rgb_array', help='Render mode: human for display, rgb_array for video')
    parser.add_argument('--render', action='store_true', help='Enable rendering for rgb_array mode (save video)')
    parser.add_argument('--print_detections', action='store_true', help='Print bullet and enemy detections')
    parser.add_argument('--clear_screen', action='store_true', help='Clear screen when no detections')

    args = parser.parse_args()
    evaluate_dqn(**vars(args))