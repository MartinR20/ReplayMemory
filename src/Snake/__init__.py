import Snake._C as _C
import numpy as np
from gym import error, spaces, utils

class _C_Snake(_C.Game):
  def __init__(self):
    super(_C_Snake, self).__init__()
    
    self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.int32)
    self.observation_space = spaces.Box(low=0, high=255, shape=(21 * 21,), dtype=np.int8) 