## Purpose
Short, actionable guidance for AI coding agents working on this DQN Atari repo. Focus on discoverable patterns, CLI workflows, file locations, and gotchas that speed up contribution.

## **Architecture Overview**
- **Policy / Target Nets**: `models/dqn_model.py` defines a 4-conv + 2-FC DQN. Inputs are normalized to [0,1] and expected shape is `(N, C, H, W)` after `permute`.
- **Environment wrapper**: `envs/atari_env.py` wraps Gym/ALE. It performs grayscale, resize to `84x84`, frame-skipping (default 4), and frame-stacking (default 4) producing observation shape `(84,84,4)`.
- **Replay & Demos**: `models/replay_buffer.py` contains `ReplayBuffer` with two memory stores: `memory` (agent) and `human_memory` (human demos). `push_human()` is used to add demo episodes; `sample(batch_size, n_step, gamma)` mixes human and agent data according to `human_ratio`.
- **Training loop**: `train.py` orchestrates training—epsilon-greedy action selection, optimizer steps, n-step return support, target network updates, TensorBoard logging, checkpointing, and plotting average rewards.

## **Key Workflows & Commands**
- **Train (resume support)**: `python train.py --env_name ALE/AirRaid-v5 --num_episodes 1000 --n_step 5 --checkpoint_path checkpoints/dqn_checkpoint_episode_400.pth --start_episode 400`
- **Evaluate / Render / Video**: `python eval.py --model_path checkpoints/dqn_checkpoint_final.pth --num_episodes 5 --render --render_mode rgb_array --save_video`
- **Collect human demos**: `python collect_human_demos.py --num_demos 10` (uses `keyboard`, optional `--use_mouse` for PID/mouse tracking; saves `human_demos.pkl`).

## **Project-Specific Conventions & Gotchas**
- **Checkpoint format**: The code supports two styles: (A) full checkpoint dict with keys `policy_net`, `target_net`, `optimizer`, `episode`, `steps_done` (preferred), and (B) older plain `state_dict`. `train.py` and `eval.py` handle both—prefer saving/loading full dicts.
- **Obs shape / model input**: Environment returns `(84,84,4)` (H,W,stack). The code always does `torch.from_numpy(state).unsqueeze(0).permute(0,3,1,2)` to get `(1,4,84,84)` before passing to the model.
- **Reward shaping & lives**: `AtariEnv.step()` applies a death penalty (`-500`) when lives drop—this is part of the observable training signal and can strongly affect learning.
- **Replay n-step semantics**: `ReplayBuffer.sample(..., n_step)` constructs n-step returns. Note next_state may be `None` if done within the n steps; reward is returned as a 1-D tensor.
- **Human-demo mixing**: Sampling reserves `int(batch_size*human_ratio)` slots for human demos; human memory is not ring-buffered (it grows), while agent memory is circular.
- **TensorBoard & plotting**: Training writes to the `logs/` directory. Average rewards are appended to `checkpoints/average_rewards.txt` and a PNG plot `average_rewards_plot.png` is generated every 100 episodes.

## **Integration Points / External Dependencies**
- ALE & ROMs: Uses `ale-py` (Gym ALE). Ensure ROMs installed (recommended: `AutoROM --accept-license`).
- System I/O: `collect_human_demos.py` relies on `keyboard`, `pyautogui`, and `cv2` (native UIs); tests may require running on a desktop session.
- Visualization/video: `imageio` used for saving evaluation videos; `matplotlib` for plotting.

## **Files to inspect first (quick start)**
- `train.py` — training loop, checkpointing, CLI args.
- `envs/atari_env.py` — preprocessing, custom detection helpers: `_get_bullets`, `_get_enemy_planes`, `_get_plane_x`.
- `models/dqn_model.py` — network architecture and normalization.
- `models/replay_buffer.py` — human demo handling and n-step logic.
- `collect_human_demos.py` & `eval.py` — CLI examples and runtime requirements.

## **Small examples / snippets agents should use**
- Normalize and permute obs before model call:
	`state = torch.from_numpy(state).unsqueeze(0).permute(0,3,1,2).to(device)`
- Load full checkpoint safely:
	```python
	ck = torch.load(path, map_location=device)
	if isinstance(ck, dict) and 'policy_net' in ck:
			policy.load_state_dict(ck['policy_net'])
	else:
			policy.load_state_dict(ck)
	```

## **When to ask the repo owner**
- If you need deterministic replay seeds or exact training hyperparameters beyond defaults in `train.py`.
- Clarify intended distribution/retention policy for human demos (how many to keep, cleaning).

If you'd like, I can (1) run a quick static pass to extract exact CLI defaults used across scripts, or (2) open a PR with this file updated and an example `DEVNOTES.md` summarizing reproducible training commands. Which would you prefer?
<parameter name="filePath">d:\rl_game_workspace_mulitistep\.github\copilot-instructions.md