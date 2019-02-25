import torch
import ReplayMemory._C as _C

mem = _C.ReplayMemory(100, 4, 3, 0.99, 0.4, 0.5, '')

for x in range(100):
    mem.append(torch.ones((84,84)), 2, 9.87, bool(x % 2))

mem.update_priorities(torch.arange((50)), torch.randn((50)) * 50)

print(mem.sample(5))