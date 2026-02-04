import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.replay_buffer import PrioritizedReplayBuffer
import torch
import numpy as np

buf = PrioritizedReplayBuffer(capacity=100, human_ratio=0.0, alpha=0.6, beta=0.4, eps=1e-6)
for i in range(20):
    s = torch.zeros((1,4,84,84), dtype=torch.uint8)
    a = torch.tensor([[0]], dtype=torch.long)
    ns = torch.zeros((1,4,84,84), dtype=torch.uint8)
    r = torch.tensor([float(i%3)], dtype=torch.float32)
    d = False
    buf.push(s,a,ns,r,d)

batch, idxs, ws = buf.sample(batch_size=8, n_step=1, gamma=0.99)
print('Sampled batch size:', len(batch))
print('Indices:', idxs)
print('Weights mean:', float(np.mean(ws)))

errs = np.abs(np.random.randn(len(idxs)))
buf.update_priorities(idxs, errs)
print('Updated max priority:', float(buf.max_priority))
