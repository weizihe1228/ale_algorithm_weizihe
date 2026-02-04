# PPO Atari AirRaid

Train a PPO agent on `ALE/AirRaid-v5` using the provided wrappers.

## Setup (Windows PowerShell)

```powershell
conda activate airraid-env
pip install -r requirements.txt
```

## Train

```powershell
python train.py --num_episodes 1000 --checkpoint_every 100 --update_steps 4096
```

Resuming from a checkpoint:

```powershell
python train.py --num_episodes 1000 --resume_path checkpoints/ppo_checkpoint_episode_100.pth
```

## Evaluate

```powershell
python eval.py --model_path checkpoints/ppo_checkpoint_episode_1000.pth --render_mode human --num_episodes 5
```

Artifacts:
- `checkpoints/ppo_checkpoint_episode_*.pth` — PPO weights
- `logs/avg_reward.txt` — CSV lines `episode,avg_reward`
- `plots/avg_reward.png` — average reward curve
