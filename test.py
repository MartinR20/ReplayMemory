import torch
import ReplayMemory._C as _C
from baseline import ReplayMemory
import time

def test(func, args):
    print(func.__name__, end=": ")
    try:
        func(*args)
        print("works")
    except:
        print("error")

def test_error(func, args):
    print(func.__name__, end=": ")
    try:
        func(*args)
        print("error")
    except:
        print("works")

def normal_run_test(size):
    mem = _C.ReplayMemory(size, 4, 3, 0.99, 0.4, 0.5, '')

    for x in range(size):
        mem.append(torch.ones((1,84,84)), 2, 9.87, bool(x % 2))

    mem.sample(32)

    mem.update_priorities(torch.arange(50, dtype=torch.int32), torch.rand(50) * 50)

    mem.sample(32)

def history_length_error(size):
    mem = _C.ReplayMemory(size, 2, 3, 0.99, 0.4, 0.5, '')
    mem.append(torch.ones((1, 84,84)), 2, 9.87, bool(x % 2))

def update_priority_error(size):
    mem = _C.ReplayMemory(size, 2, 3, 0.99, 0.4, 0.5, '')

    for x in range(size):
        mem.append(torch.ones((1,84,84)), 2, 9.87, bool(x % 2))

    mem.update_priorities(torch.arange(10, dtype=torch.int32), torch.rand(9) * 50)

def iteration_test(size):
    mem = _C.ReplayMemory(size, 2, 3, 0.99, 0.4, 0.5, '')

    for x in range(size):
        mem.append(torch.ones((1,84,84)), 2, 9.87, bool(x % 2))
    
    for x in mem: pass
    for x in mem: pass

size = 50000
test(normal_run_test, (size,))
test_error(history_length_error, (size,))
test_error(update_priority_error, (size,))
test(iteration_test, (size,))
