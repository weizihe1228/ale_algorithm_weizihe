# DQN for Atari AirRaid-v5

This project implements a Deep Q-Network (DQN) algorithm using PyTorch to play the Atari game AirRaid-v5 via the ALE (Arcade Learning Environment).

## Project Structure

- `envs/atari_env.py`: Atari environment wrapper with preprocessing (grayscale, resize, frame stacking)
- `models/dqn_model.py`: DQN neural network model
- `models/replay_buffer.py`: Experience replay buffer
- `train.py`: Training script
- `eval.py`: Evaluation script
- `visualize.py`: Visualization script for trained agent
- `requirements.txt`: Python dependencies

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure ROMs are installed (using AutoROM):
   ```bash
   AutoROM --accept-license
   ```

## Usage

### Training
Run the training script:
```bash
python train.py
```

### Evaluation
Evaluate a trained model:
```bash
python eval.py
```

### Visualization
Generate a video of the agent playing:
```bash
python visualize.py
```

## Hyperparameters

- Learning rate: 1e-4
- Batch size: 32
- Gamma: 0.99
- Epsilon start/end: 1.0 / 0.01
- Epsilon decay: 1000
- Target network update: every 1000 steps
- Replay buffer capacity: 10000

## Notes

- The environment uses frame skipping (4 frames), frame stacking (4 frames), and resizes to 84x84 grayscale.
- Checkpoints are saved every 100 episodes in the `checkpoints/` directory.
- Training logs are written to TensorBoard in the `logs/` directory.