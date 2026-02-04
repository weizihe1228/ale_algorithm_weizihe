import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def _init_layer(layer: nn.Module):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class CNNPolicy(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        # Compute flattened size for 84x84 input
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.features(dummy).reshape(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
        )
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

        # Init weights
        for m in self.modules():
            _init_layer(m)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.features(x)
        # Use reshape to handle possibly non-contiguous tensors (safer than view)
        z = z.reshape(z.size(0), -1)
        z = self.fc(z)
        logits = self.actor(z)
        value = self.critic(z)
        return logits, value.squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-5
    update_epochs: int = 2
    batch_size: int = 128
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class PPOAgent:
    def __init__(self, in_channels: int, num_actions: int, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.policy = CNNPolicy(in_channels, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    def act(self, obs_np: np.ndarray) -> Tuple[int, float, float]:
        # obs_np: HWC uint8 stack; convert to CHW float in [0,1]
        obs_chw = np.transpose(obs_np, (2, 0, 1))
        obs_t = torch.from_numpy(obs_chw).float().div(255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def evaluate_actions(self, obs_batch: torch.Tensor, actions: torch.Tensor):
        logits, values = self.policy(obs_batch)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy, values

    def compute_gae(self, rewards, dones, values, next_value):
        cfg = self.config
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + cfg.gamma * (1.0 - dones[t]) * (next_value if t == len(rewards) - 1 else values[t + 1]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [advantages[i] + values[i] for i in range(len(values))]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(self, trajectories: List[dict]):
        # Flatten
        obs_list, act_list, logp_list, val_list, adv_list, ret_list = [], [], [], [], [], []
        for traj in trajectories:
            obs_list += traj['obs']
            act_list += traj['actions']
            logp_list += traj['logprobs']
            val_list += traj['values']
            adv_t, ret_t = self.compute_gae(traj['rewards'], traj['dones'], traj['values'], traj['next_value'])
            adv_list += adv_t.tolist()
            ret_list += ret_t.tolist()

        # Normalize advantages
        adv = torch.tensor(adv_list, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Build tensors
        obs_np = np.stack(obs_list).astype(np.uint8)
        obs_chw = np.transpose(obs_np, (0, 3, 1, 2))
        obs_t = torch.from_numpy(obs_chw).float().div(255.0).to(self.device)
        actions = torch.tensor(act_list, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(logp_list, dtype=torch.float32).to(self.device)
        returns = torch.tensor(ret_list, dtype=torch.float32).to(self.device)
        advantages = adv.to(self.device)

        cfg = self.config
        n_steps = obs_t.size(0)
        idxs = np.arange(n_steps)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n_steps, cfg.batch_size):
                end = start + cfg.batch_size
                mb = idxs[start:end]
                mb_obs = obs_t[mb]
                mb_actions = actions[mb]
                mb_old_logp = old_logprobs[mb]
                mb_returns = returns[mb]
                mb_advantages = advantages[mb]

                new_logp, entropy, values = self.evaluate_actions(mb_obs, mb_actions)
                ratio = (new_logp - mb_old_logp).exp()
                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (mb_returns - values).pow(2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

        return {
            'policy_loss': float(pg_loss.item()),
            'value_loss': float(v_loss.item()),
            'entropy': float(ent_loss.item()),
            'n_steps': n_steps,
        }

    def save(self, path: str, episode: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'episode': episode,
            'model_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.__dict__,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        return int(ckpt.get('episode', 0))
