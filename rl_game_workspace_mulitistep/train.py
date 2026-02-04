import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from envs.atari_env import AtariEnv
from models.dqn_model import DQN
from models.replay_buffer import ReplayBuffer, Transition
import os
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt

def train_dqn(env_name='ALE/AirRaid-v5', num_episodes=10000, batch_size=128, gamma=0.994,
              eps_start=1.0, eps_end=0.05, eps_decay=1000, target_update=1000,
              replay_capacity=10000, learning_rate=1e-5, save_path='checkpoints/',
              log_path='logs/', checkpoint_path=None, start_episode=0, human_demo_path=None, human_ratio=0.5, n_step=1, resume_eps=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    avg_file = os.path.join(save_path, 'average_rewards.txt')

    env = AtariEnv(env_name)
    num_actions = env.action_space.n

    policy_net = DQN((4, 84, 84), num_actions).to(device)
    target_net = DQN((4, 84, 84), num_actions).to(device)
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if it's a full checkpoint or just state_dict
        if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            # Full checkpoint
            policy_net.load_state_dict(checkpoint['policy_net'])
            target_net.load_state_dict(checkpoint['target_net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_episode = checkpoint.get('episode', 0)
            steps_done = checkpoint.get('steps_done', 0)
            print(f"Resuming from episode {start_episode}, steps done: {steps_done}")
        else:
            # Old checkpoint (only state_dict)
            policy_net.load_state_dict(checkpoint)
            target_net.load_state_dict(policy_net.state_dict())
            print("Loaded old checkpoint (state_dict only), starting from episode 0")
    else:
        target_net.load_state_dict(policy_net.state_dict())
    
    target_net.eval()
    memory = ReplayBuffer(replay_capacity, human_ratio=human_ratio)

    writer = SummaryWriter(log_path)

    steps_done = 0
    rewards = []  # Store episode rewards
    # No separate resume counter needed: saving will be based on start_episode

    current_eps = None  # For manual epsilon override on resume
    if checkpoint_path and resume_eps is not None:
        current_eps = resume_eps

    def select_action(state, eps_threshold):
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)

    print(f"Starting training from episode {start_episode}, running {num_episodes} episodes", flush=True)
    for episode in range(start_episode, start_episode + num_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        total_reward = 0
        done = False

        while not done:
            if current_eps is not None:
                eps_threshold = current_eps
            else:
                eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-steps_done / eps_decay)
            action = select_action(state, eps_threshold)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            reward = torch.tensor([reward], device=device)
            next_state = torch.from_numpy(next_state).unsqueeze(0).permute(0, 3, 1, 2).to(device) if not done else None

            memory.push(state, action, next_state, reward, done)

            state = next_state
            total_reward += reward.item()
            steps_done += 1

            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size, n_step, gamma)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states_list = [s for s in batch.next_state if s is not None]

                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                # Ensure reward_batch is 1-D tensor of shape (batch,)
                reward_batch = torch.cat(batch.reward).squeeze()

                state_action_values = policy_net(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(batch_size, device=device)
                if len(non_final_next_states_list) > 0:
                    non_final_next_states = torch.cat(non_final_next_states_list)
                    with torch.no_grad():
                        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

                # expected_state_action_values is 1-D (batch,), unsqueeze to (batch,1) for loss
                expected_state_action_values = reward_batch + (gamma ** n_step) * next_state_values

                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

                # TensorBoard logging for training diagnostics
                try:
                    writer.add_scalar('Train/loss', loss.item(), steps_done)
                    writer.add_scalar('Train/epsilon', eps_threshold, steps_done)
                    writer.add_scalar('Train/memory_size', len(memory), steps_done)
                except Exception:
                    pass

            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        writer.add_scalar('Reward/train', total_reward, episode)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {eps_threshold:.3f}", flush=True)
        rewards.append(total_reward)

        # Save checkpoints every 100 episodes starting from `start_episode`.
        # This means saves occur at: start_episode, start_episode+100, start_episode+200, ...
        if (episode - start_episode) % 100 == 0:
            checkpoint = {
                'episode': episode,
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'steps_done': steps_done
            }
            torch.save(checkpoint, os.path.join(save_path, f'dqn_checkpoint_episode_{episode}.pth'))
            print(f"Saved checkpoint at episode {episode} -> {os.path.join(save_path, f'dqn_checkpoint_episode_{episode}.pth')}", flush=True)

            # Calculate and save average reward
            if len(rewards) >= 10:
                avg_reward = sum(rewards[-10:]) / 10  # Average of last 10 episodes
                # append average to file inside save_path
                with open(avg_file, 'a') as f:
                    f.write(f"{episode},{avg_reward}\n")
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass

                # Log average reward (last 10) to TensorBoard
                try:
                    writer.add_scalar('Reward/avg10', avg_reward, episode)
                except Exception:
                    pass

                # Plot
                episodes_list = []
                avgs = []
                # read from the average file inside save_path
                if os.path.exists(avg_file):
                    with open(avg_file, 'r') as f:
                        for line in f:
                            s = line.strip()
                            if not s:
                                continue
                            parts = s.split(',')
                            if len(parts) != 2:
                                # skip malformed lines
                                continue
                            ep, avg = parts
                            try:
                                episodes_list.append(int(ep))
                                avgs.append(float(avg))
                            except Exception:
                                # skip lines that don't parse to numbers
                                continue
                else:
                    # no data yet
                    episodes_list = []
                    avgs = []

                plt.figure(figsize=(10, 5))
                # draw lines with markers for visibility
                if len(episodes_list) > 0:
                    plt.plot(episodes_list, avgs, 'o-')
                else:
                    # empty plot with labels
                    plt.plot([], [])
                plt.xlabel('Episode')
                plt.ylabel('Average Reward (last 10)')
                plt.title('Average Reward per 100 Episodes')
                plt.grid(True)
                plot_path = os.path.join(save_path, 'average_rewards_plot.png')
                plt.savefig(plot_path)
                plt.close()

                print(f"Saved average reward plot at episode {episode} -> {plot_path}", flush=True)

    env.close()
    writer.close()
    
    # Save final checkpoint
    final_checkpoint = {
        'episode': episode,
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'steps_done': steps_done
    }
    torch.save(final_checkpoint, os.path.join(save_path, f'dqn_checkpoint_final.pth'))
    print(f"Final checkpoint saved at episode {episode}")
    
    print("Training completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train DQN on Atari games')
    parser.add_argument('--env_name', type=str, default='ALE/AirRaid-v5', help='Environment name')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Starting epsilon')
    parser.add_argument('--eps_end', type=float, default=0.01, help='Ending epsilon')
    parser.add_argument('--eps_decay', type=float, default=1000, help='Epsilon decay')
    parser.add_argument('--target_update', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--replay_capacity', type=int, default=10000, help='Replay buffer capacity')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='checkpoints/', help='Path to save checkpoints')
    parser.add_argument('--log_path', type=str, default='logs/', help='Path to save logs')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to load checkpoint')
    parser.add_argument('--start_episode', type=int, default=0, help='Starting episode number')
    parser.add_argument('--human_demo_path', type=str, default=None, help='Path to human demonstration file')
    parser.add_argument('--human_ratio', type=float, default=0.5, help='Ratio of human data in batch sampling')
    parser.add_argument('--n_step', type=int, default=5, help='Number of steps for n-step Q-learning')
    parser.add_argument('--resume_eps', type=float, default=0.02, help='Manual epsilon value when resuming from checkpoint (overrides decay)')

    args = parser.parse_args()
    train_dqn(**vars(args))