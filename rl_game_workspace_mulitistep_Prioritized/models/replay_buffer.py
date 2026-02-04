import numpy as np
import random
from collections import namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, human_ratio=0.5):
        self.capacity = capacity
        self.memory = []
        self.human_memory = []
        self.position = 0
        self.human_ratio = human_ratio  # 人类数据的采样比例

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def push_human(self, *args):
        # 人类数据不循环覆盖，保持固定
        self.human_memory.append(Transition(*args))

    def sample(self, batch_size, n_step=1, gamma=0.99):
        human_batch_size = int(batch_size * self.human_ratio)
        robot_batch_size = batch_size - human_batch_size

        batch = []
        if len(self.human_memory) > 0 and human_batch_size > 0:
            batch.extend(random.sample(self.human_memory, min(human_batch_size, len(self.human_memory))))
        if len(self.memory) > 0 and robot_batch_size > 0:
            if n_step == 1:
                batch.extend(random.sample(self.memory, min(robot_batch_size, len(self.memory))))
            else:
                for _ in range(robot_batch_size):
                    if len(self.memory) >= n_step:
                        idx = random.randint(0, len(self.memory) - n_step)
                        state = self.memory[idx].state
                        action = self.memory[idx].action
                        n_step_return = 0.0
                        discount = 1.0
                        done = False
                        next_state = None
                        for i in range(n_step):
                            trans = self.memory[idx + i]
                            reward_val = trans.reward.item() if hasattr(trans.reward, 'item') else trans.reward
                            n_step_return += discount * reward_val
                            discount *= gamma
                            if trans.done:
                                done = True
                                break
                        if not done:
                            next_state = self.memory[idx + n_step - 1].next_state
                        batch.append(Transition(state, action, next_state, torch.tensor([n_step_return], device=state.device if hasattr(state, 'device') else 'cpu'), done))
        # 如果不够，填充机器人数据
        while len(batch) < batch_size and len(self.memory) > 0:
            batch.extend(random.sample(self.memory, 1))

        return batch[:batch_size]

    def __len__(self):
        return len(self.memory) + len(self.human_memory)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, human_ratio=0.5, alpha=0.6, beta=0.4, eps=1e-6):
        self.capacity = capacity
        self.memory = []
        self.human_memory = []
        self.position = 0
        self.human_ratio = human_ratio
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # New transitions get max priority to ensure they are sampled soon
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def push_human(self, *args):
        self.human_memory.append(Transition(*args))

    def sample(self, batch_size, n_step=1, gamma=0.99):
        human_batch_size = int(batch_size * self.human_ratio)
        robot_batch_size = batch_size - human_batch_size

        batch = []
        indices = []
        weights = []

        # Human data sampled uniformly as-is
        if len(self.human_memory) > 0 and human_batch_size > 0:
            batch.extend(random.sample(self.human_memory, min(human_batch_size, len(self.human_memory))))

        mem_len = len(self.memory)
        if mem_len > 0 and robot_batch_size > 0:
            # Determine valid indices for n-step
            if n_step > 1:
                valid_indices = np.arange(0, max(mem_len - n_step + 1, 0))
            else:
                valid_indices = np.arange(0, mem_len)

            if valid_indices.size > 0:
                probs_src = self.priorities[valid_indices]
                # Avoid all-zero priorities
                probs_src = np.where(probs_src > 0, probs_src, self.eps)
                probs = probs_src ** self.alpha
                probs_sum = probs.sum()
                if probs_sum == 0:
                    probs = np.ones_like(probs) / probs.size
                else:
                    probs = probs / probs_sum

                k = min(robot_batch_size, valid_indices.size)
                chosen = np.random.choice(valid_indices, size=k, replace=False, p=probs)

                # Importance-sampling weights
                N = valid_indices.size
                chosen_probs = probs[np.isin(valid_indices, chosen)]
                w = (N * chosen_probs) ** (-self.beta)
                w = w / (w.max() + 1e-8)

                for idx_i, w_i in zip(chosen.tolist(), w.tolist()):
                    # Build n-step transition starting from idx_i
                    state = self.memory[idx_i].state
                    action = self.memory[idx_i].action
                    n_step_return = 0.0
                    discount = 1.0
                    done = False
                    next_state = None
                    horizon = n_step if (idx_i + n_step) <= mem_len else (mem_len - idx_i)
                    for i in range(horizon):
                        trans = self.memory[idx_i + i]
                        reward_val = trans.reward.item() if hasattr(trans.reward, 'item') else trans.reward
                        n_step_return += discount * reward_val
                        discount *= gamma
                        if trans.done:
                            done = True
                            break
                    if not done and horizon >= 1:
                        target_idx = idx_i + horizon - 1
                        next_state = self.memory[target_idx].next_state
                    batch.append(Transition(state, action, next_state, torch.tensor([n_step_return], device=state.device if hasattr(state, 'device') else 'cpu'), done))
                    indices.append(idx_i)
                    weights.append(w_i)

        # If not enough, pad robot samples uniformly
        while len(batch) < batch_size and mem_len > 0:
            idx_i = random.randrange(mem_len)
            trans = self.memory[idx_i]
            batch.append(trans)
            indices.append(idx_i)
            weights.append(1.0)

        # Ensure sizes align
        if len(indices) == 0:
            indices = [0] * len(batch)
            weights = [1.0] * len(batch)

        return batch[:batch_size], indices[:batch_size], np.array(weights[:batch_size], dtype=np.float32)

    def update_priorities(self, indices, td_errors):
        # td_errors: numpy array or list of absolute TD errors
        for idx, err in zip(indices, td_errors):
            pr = float(abs(err)) + self.eps
            self.priorities[idx] = pr
            if pr > self.max_priority:
                self.max_priority = pr

    def __len__(self):
        return len(self.memory) + len(self.human_memory)