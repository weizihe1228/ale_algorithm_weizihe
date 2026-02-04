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