import numpy as np
import random
from collections import namedtuple

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

    def sample(self, batch_size):
        human_batch_size = int(batch_size * self.human_ratio)
        robot_batch_size = batch_size - human_batch_size

        batch = []
        if len(self.human_memory) > 0 and human_batch_size > 0:
            batch.extend(random.sample(self.human_memory, min(human_batch_size, len(self.human_memory))))
        if len(self.memory) > 0 and robot_batch_size > 0:
            batch.extend(random.sample(self.memory, min(robot_batch_size, len(self.memory))))

        # 如果不够，填充机器人数据
        while len(batch) < batch_size and len(self.memory) > 0:
            batch.extend(random.sample(self.memory, 1))

        return batch[:batch_size]

    def __len__(self):
        return len(self.memory) + len(self.human_memory)