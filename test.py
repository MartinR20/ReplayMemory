import torch
import ReplayMemory._C as _C
from baseline import ReplayMemory
import time

def timeit(func, args):
    start = time.time()
    func(*args)
    print(time.time() - start)

def c_bench(size):
    mem = _C.ReplayMemory(size, 4, 3, 0.99, 0.4, 0.5, '')

    for x in range(size):
        mem.append(torch.ones((1, 84,84)), 2, 9.87, bool(x % 2))

    rang = int(size*2/3)
    mem.update_priorities(torch.arange(rang), torch.rand(rang) * 50)

    mem.sample(32)

def baseline_bench(size):
    mem = ReplayMemory(size, 4, 3, 0.99, 0.4, 0.5, 'cpu')

    for x in range(size):
        mem.append(torch.ones((1,84,84)), 2, 9.87, bool(x % 2))

    rang = int(size*2/3)
    mem.update_priorities(torch.arange(rang), (torch.rand(rang) * 50).numpy())

    mem.sample(32)

size = 50000
timeit(c_bench, (size,))
timeit(baseline_bench, (size,))
