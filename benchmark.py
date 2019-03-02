import torch
import ReplayMemory._C as _C
from baseline import ReplayMemory
import time

def timeit(func, args):
    start = time.time()
    ret = func(*args)
    print(time.time() - start)

    return ret

def c_bench(size):
    mem = _C.ReplayMemory(size, 4, 3, 0.99, 0.4, 0.5, '')

    for x in range(size):
        mem.append(torch.ones((1, 84,84)), 2, 9.87, bool(x % 2))

    rang = int(size*2/3)
    mem.update_priorities(torch.arange(rang, dtype=torch.int32), torch.rand(rang) * 50)

    mem.sample(32)

    return mem

def baseline_bench(size):
    mem = ReplayMemory(size, 4, 3, 0.99, 0.4, 0.5, 'cpu')

    for x in range(size):
        mem.append(torch.ones((1,84,84)), 2, 9.87, bool(x % 2))

    rang = int(size*2/3)
    mem.update_priorities(torch.arange(rang), (torch.rand(rang) * 50).numpy())

    mem.sample(32)
    
    return mem

def sample_bench(mem, batch_size, its):
    for x in range(its):
        mem.sample(batch_size)

size = 10000
batch_size = 32
samples = 50

c_mem = timeit(c_bench, (size,))
baseline_mem = timeit(baseline_bench, (size,))

timeit(sample_bench, (c_mem, batch_size, samples))
timeit(sample_bench, (baseline_mem, batch_size, samples))
