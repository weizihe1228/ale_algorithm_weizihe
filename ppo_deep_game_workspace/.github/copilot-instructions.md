## Repository guide for AI coding agents

This guide makes AI agents immediately productive in this PPO Atari project. It summarizes architecture, conventions, and the exact workflows used here.

- **Big picture:** Train and evaluate a PPO agent on `ALE/AirRaid-v5`. The Atari wrappers in `envs/` handle preprocessing, frame skip/stack, and reward shaping. The PPO implementation lives in `ppo_agent.py`; `train.py` orchestrates data collection and PPO updates; `eval.py`/`eval_deterministic.py` run evaluation.

- **Key files:**
  - `envs/atari_env.py` — Single-env wrapper. HWC observations `(84,84,stack)`, `gym.make(..., render_mode=...)`, OpenCV grayscale+resize, life tracking via `info.get('lives', ...)`. Death penalty is `-50` here.
  - `envs/atari_env_vector.py` — Vector/TimeLimit wrapper. CHW observations `(stack,84,84)`, `gym.wrappers.TimeLimit(gym.make(...))`, lives via `self.env.unwrapped.ale.lives()`. Death penalty is `-500` here.
  - `ppo_agent.py` — `CNNPolicy` + `PPOAgent`. Converts HWC uint8 to CHW float `[0,1]`, samples actions via `Categorical(logits=...)`, computes GAE, and runs clipped PPO updates.
  - `train.py` — Collects trajectories, triggers updates every `--update_steps`, logs avg reward to `logs/avg_reward.txt`, saves plot `plots/avg_reward.png` every 10 episodes, checkpoints under `checkpoints/`.
  - `eval.py` — Stochastic eval using `PPOAgent.load(...)` and `agent.act(...)` with optional human rendering.
  - `eval_deterministic.py` — Deterministic eval: loads `model_state` and selects `argmax(logits)` after explicit HWC→CHW conversion.
  - `README.md` — Canonical commands for setup/train/eval; prefer these over legacy snippets in `envs/start.txt`.

- **Architecture & conventions:**
  - Observation layout is deliberate: `atari_env.py` returns HWC; `ppo_agent.py` and `eval_deterministic.py` convert to CHW for the CNN. Do not change axis order without updating the agent and scripts.
  - Dtype/range: observations are `uint8` `[0,255]` from wrappers; the agent normalizes to float `[0,1]`. Keep wrapper `observation_space=Box(0,255, dtype=uint8)` unless updating all consumers.
  - Reward shaping: life-loss penalties differ (`-50` single-env vs `-500` vector). Changing these impacts stability; align changes across training/eval if you modify them.
  - Rendering: training uses `render_mode='rgb_array'` to avoid GUI overhead; eval can use `human`. Vector wrapper omits `render_mode` for compatibility.
  - Checkpoints: `PPOAgent.save` stores `episode`, `model_state`, `optimizer_state`, and `config`. `eval_deterministic.py` loads only `model_state` and reads `episode` for logging.

- **Developer workflows (Windows PowerShell):**
  - Setup:
```powershell
conda activate airraid-env
pip install -r requirements.txt
```
  - Train (PPO):
```powershell
python train.py --num_episodes 1000 --checkpoint_every 100 --update_steps 4096
```
  - Resume:
```powershell
python train.py --num_episodes 1000 --resume_path checkpoints/ppo_checkpoint_episode_100.pth
```
  - Evaluate (stochastic):
```powershell
python eval.py --model_path checkpoints/ppo_checkpoint_episode_1000.pth --render_mode human --num_episodes 5
```
  - Evaluate (deterministic):
```powershell
python eval_deterministic.py --model_path checkpoints/ppo_checkpoint_episode_1000.pth --render_mode human --num_episodes 5
```

- **Practical tips & pitfalls:**
  - Shape checks: confirm `env.observation_space.shape == (84,84,4)` in single-env; the agent internally transposes to `(4,84,84)`.
  - Life tracking: single-env uses `info['lives']`; vector-env uses `env.unwrapped.ale.lives()`. Keep consistent if you port logic.
  - Update cadence: `train.py` prints `[DEBUG]` lines when `steps_since_update >= --update_steps`; use these to verify trajectory sizing.
  - Plot/logs: `logs/avg_reward.txt` appends `episode,avg_reward`; plots saved every 10 episodes. Avoid heavy plotting per-episode.
  - `start.txt` contains older DQN commands; use `README.md` for current PPO commands.

- **Search anchors:** `frame_stack`, `frame_skip`, `screen_size`, `PPOAgent`, `CNNPolicy`, `update_steps`, `avg_reward.txt`, `TimeLimit`, `lives()`.

If anything here is unclear or missing (e.g., model I/O specifics, expected observation layouts for new models, or a checklist for refactoring wrappers), tell me what to expand and I’ll iterate.
