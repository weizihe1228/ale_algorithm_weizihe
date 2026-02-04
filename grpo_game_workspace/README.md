# GRPO Atari AirRaid

Train a GRPO agent on `ALE/AirRaid-v5` using the provided wrappers.

## Setup (Windows PowerShell)

```powershell
conda activate airraid-env
pip install -r requirements.txt
```

## Train

```powershell
python train.py --num_steps 200000 --rollout_steps 2048 --checkpoint_every 100
```

Resuming from a checkpoint:

```powershell
python train.py --resume_path checkpoints/grpo_checkpoint_episode_0100.pth --num_steps 200000
```

## Evaluate

```powershell
python eval.py --model_path checkpoints/grpo_checkpoint_episode_0100.pth --render_mode human --num_episodes 5
```

Artifacts:
- `checkpoints/grpo_checkpoint_episode_*.pth` — GRPO weights
- `logs/avg_reward.txt` — CSV lines `episode,avg_reward`
- `plots/avg_reward.png` — average reward curve
