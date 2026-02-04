## Repository guide for AI coding agents

This file contains concise, project-specific guidance for code changes and investigation. Focus on the `envs/` Atari wrappers and the training/eval scripts referenced in `start.txt`.

- **Big picture:** The repo provides Atari environment wrappers used for RL experiments. Primary code lives in `envs/atari_env.py` (single-env wrapper) and `envs/atari_env_vector.py` (vector/TimeLimit wrapper). These files implement preprocessing, frame-skipping/stacking, and reward shaping (heavy life-loss penalty).

- **Key files:**
  - `envs/atari_env.py` — HWC observation (shape `(H,W,frames)`), uses `gym.make(..., render_mode=...)`, OpenCV-based preprocessing, and `info.get('lives', ...)` for life tracking.
  - `envs/atari_env_vector.py` — CHW observation (shape `(frames,H,W)`), uses `gym.wrappers.TimeLimit(gym.make(...))`, accesses lives with `self.env.unwrapped.ale.lives()`.
  - `start.txt` — contains example PowerShell/conda run commands used to evaluate/train models (useful for reproducing runs locally).

- **Architectural notes / why things are structured this way:**
  - Two wrappers exist because single-env workflows (rendering, info dict usage) differ from vectorized/time-limited setups. Keep changes consistent with the wrapper being edited.
  - Observation shapes differ intentionally: `atari_env.py` uses HWC (height,width,stack) while `atari_env_vector.py` uses CHW (stack,height,width). When editing code that consumes observations (models, augmentations, dataloaders), confirm the expected layout.
  - Preprocessing strategies differ: `atari_env.py` uses `cv2` grayscale+resize; `atari_env_vector.py` uses a manual nearest-neighbor resize loop. Maintain dtype `uint8` and range `0-255` unless you update all downstream consumers.

- **Important behaviors to preserve when modifying reward/observation logic:**
  - Life loss penalty: both wrappers apply a large negative reward on losing a life (example: `-500`). This is a global shaping decision — changing it affects training stability.
  - Frame stack ordering: code `self.frames.append(frame)` and `np.stack(..., axis=...)` define temporal ordering. Keep consistent ordering and note which axis is temporal for any model input.

- **Common modification pitfalls:**
  - Mixing HWC/CHW: don't change the axis order in one file without updating all consumers (models, logging, tests).
  - Changing dtype or scaling: observation_space is `Box(0,255, dtype=uint8)` — if you normalize to float, update the model and any saved checkpoints accordingly.
  - Render mode / TimeLimit: `atari_env.py` passes `render_mode` to `gym.make`, while `atari_env_vector.py` removes `render_mode` for vector compatibility. If you add rendering to vector envs, ensure compatibility.

- **Searchable anchors in codebase (use these to find patterns quickly):**
  - `frame_stack`, `frame_skip`, `screen_size` — parameters controlling preprocessing.
  - `lives()` and `info.get('lives'` — two different life-tracking approaches.
  - `bullet` / `plane` — commented detection/reward shaping code exists; useful examples of visual heuristics.

- **Developer workflows / commands (PowerShell / Windows):**
  - Eval using conda env (example from `start.txt`):
```powershell
conda run -p C:\Users\weizi\.conda\envs\airraid-env --no-capture-output python eval.py --model_path checkpoints/dqn_checkpoint_episode_2700.pth --render_mode human
```
  - Train (example):
```powershell
python train.py --num_episodes 10000 --batch_size 32 --learning_rate 1e-4 --eps_decay 100000 --replay_capacity 100000 --target_update 1000 --n_step 3
```

- **Testing & debugging tips:**
  - Small-scale sanity: run a few episodes with `render_mode='rgb_array'` or `human` depending on wrapper, confirm obs shapes and ranges.
  - Instrument life tracking: print `info` in `atari_env.py` and check `self.env.unwrapped.ale.lives()` in the vector env to confirm they match for the chosen ROM.
  - When refactoring vision code, keep one working implementation (do not replace both resizing methods at once).

- **When you see commented-out visual heuristics:** the repo contains experimental code for bullet / plane detection. Treat these as examples (not maintained production code). If re-enabling, add unit tests or an evaluation script that logs detection counts for several episodes.

If anything in this guidance is unclear or you'd like more examples (model I/O examples, training config files, or a checklist for a PR touching envs), tell me which area to expand and I'll iterate.
