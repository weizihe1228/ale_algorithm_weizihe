# AI Coding Agent Instructions for DQN Atari Project

## Architecture Overview
This is a PyTorch-based Deep Q-Network (DQN) implementation for Atari AirRaid-v5 with n-step Q-learning support. Core components:
- `envs/atari_env.py`: Gymnasium wrapper with frame preprocessing (grayscale, 84x84 resize, 4-frame stacking/skip)
- `models/dqn_model.py`: 4-conv + 2-FC DQN network
- `models/replay_buffer.py`: Experience replay with human demo mixing and n-step return calculation
- `train.py`: Main training loop with epsilon-greedy, target network updates, TensorBoard logging, n-step TD targets

## Key Workflows
- **Training**: `python train.py --env_name ALE/AirRaid-v5 --num_episodes 1000 --n_step 3 --checkpoint_path checkpoints/dqn_checkpoint_episode_400.pth --start_episode 400`
- **Evaluation**: `python eval.py --model_path checkpoints/dqn_checkpoint_final.pth --num_episodes 10 --save_video`
- **Visualization**: `python visualize.py --model_path checkpoints/dqn_checkpoint_final.pth`
- **Human Demos**: `python collect_human_demos.py --num_demos 10` (uses keyboard/mouse input)

## Patterns & Conventions
- **Checkpoint Format**: Save/load full dict with 'policy_net', 'target_net', 'optimizer', 'episode', 'steps_done' (see `train.py:110-120`)
- **Frame Preprocessing**: Always permute to (N, C, H, W) and normalize to [0,1] in model forward (see `models/dqn_model.py:28`)
- **Human Demo Integration**: Use `ReplayBuffer.push_human()` for demos, sampled via `human_ratio` in `sample()` (see `models/replay_buffer.py:15-30`)
- **N-Step Returns**: `ReplayBuffer.sample()` computes n-step discounted returns when n_step > 1 (see `models/replay_buffer.py:20-40`)
- **Environment Reset/Step**: Returns stacked frames (84,84,4), handles frame skipping internally (see `envs/atari_env.py:85-110`)
- **Logging**: TensorBoard scalars for rewards, checkpoints every 100 episodes, average reward plots (see `train.py:130-170`)

## Dependencies & Setup
- Install: `pip install -r requirements.txt` + `AutoROM --accept-license`
- Device: Auto-detects CUDA, falls back to CPU
- ROMs: Requires ALE/AirRaid-v5 via ale-py

## Common Tasks
- Resume training: Load checkpoint, set `--start_episode` to resume episode count
- Add features: Extend `AtariEnv` for custom rewards/observations, update `DQN` for new architectures
- Debug: Use `eval.py --render` to visualize agent behavior, check TensorBoard logs</content>
<parameter name="filePath">d:\rl_game_workspace_mulitistep\.github\copilot-instructions.md