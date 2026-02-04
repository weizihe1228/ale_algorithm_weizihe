# AI Coding Agent Instructions for DQN Atari Project

## Architecture Overview
- `envs/atari_env.py`: Gymnasium ALE wrapper with preprocessing (grayscale, 84x84 resize, 4-frame stack/skip) and reward shaping (life-loss penalty).
- `envs/atari_env_vector.py`: Vector-friendly variant returning `(4,84,84)` arrays with simple resize.
- `models/dqn_model.py`: DQN with 4 conv layers + 2 FC; forward normalizes inputs by `/255.0` and expects `(N,C,H,W)`.
- `models/replay_buffer.py`: Experience replay with human demo mixing (`push_human`) and optional n-step return computation in `sample()`.
- `train.py`: Main loop with epsilon-greedy, target network sync, TensorBoard logging, checkpointing, optional n-step TD targets.
- `eval.py`/`visualize.py`: Load checkpoints, run greedy policy, optional video saving and simple diagnostics.

## Key Workflows
- Training (defaults from `argparse`):
	- `python train.py --env_name ALE/AirRaid-v5 --num_episodes 1000 --batch_size 128 --gamma 0.99 --eps_decay 1000 --target_update 1000 --replay_capacity 10000 --learning_rate 1e-4 --save_path checkpoints --log_path logs --n_step 5`
- Training with PER:
	- `python train.py --use_per --per_alpha 0.6 --per_beta 0.4 --per_eps 1e-6`
- Resume training from checkpoint:
	- `python train.py --checkpoint_path checkpoints/dqn_checkpoint_episode_400.pth --start_episode 400 --resume_eps 0.02`
- Evaluation (accepts full or state_dict checkpoints):
	- `python eval.py --model_path checkpoints/dqn_checkpoint_final.pth --num_episodes 5 --render_mode rgb_array --render`
- Visualization (human/rgb_array):
	- `python visualize.py --model_path checkpoints/dqn_checkpoint_episode_900.pth --render_mode human`
- Human demos collection (keyboard or `--use_mouse`):
	- `python collect_human_demos.py --use_mouse`

## Patterns & Conventions
- Observations: env returns `(84,84,4)` uint8; convert to tensor and `permute(0,3,1,2)` before model; model normalizes in `forward`.
- Actions: epsilon schedule `eps_end + (eps_start-eps_end) * exp(-steps_done/eps_decay)`; `select_action` returns argmax when exploiting.
- Target sync: copy policy → target every `--target_update` steps.
- ReplayBuffer: samples mix in human data by `human_ratio`; for `n_step>1`, computes discounted return over up to n transitions and sets `next_state` at step `n-1` unless done.
- PER: when `--use_per`, sample returns `(batch, indices, weights)`; loss uses per-sample Huber weighted by IS `weights`, priorities updated via absolute TD error.
- Checkpoints: saved every 100 episodes starting at `--start_episode`; full dict with `policy_net`, `target_net`, `optimizer`, `episode`, `steps_done`.
- Logging: TensorBoard scalars for loss/epsilon/memory; average of last 10 rewards appended to `checkpoints/average_rewards.txt` and plotted to `average_rewards_plot.png`.
- Reward shaping: life loss triggers `-500` penalty in `AtariEnv.step`; bullet/enemy penalties are present but commented.

## Dependencies & Setup
- Install: `pip install -r requirements.txt`; install ROMs: `AutoROM --accept-license`.
- Devices: auto-detects CUDA, falls back to CPU.
- ALE env: use `--env_name ALE/AirRaid-v5`; Gymnasium + `ale-py` required.

## Integration Notes
- Checkpoint loading: both full dict and raw `state_dict` are supported in `eval.py`/`visualize.py`.
- Human demos: `collect_human_demos.py` saves a pickle; to train with demos, load and push via `ReplayBuffer.push_human`. `--human_demo_path` exists in `train.py` but demo loading is not wired yet.

## File Pointers
- Training loop and checkpointing: `train.py`.
- Preprocessing & reward shaping: `envs/atari_env.py`.
- DQN architecture: `models/dqn_model.py`.
- Replay/n-step/human mixing: `models/replay_buffer.py`.
- Eval/video: `eval.py`, `visualize.py`.