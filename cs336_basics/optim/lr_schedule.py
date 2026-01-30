import math
from typing import Optional

class cos_schedule():
    def __init__(self, lr_max: float, lr_min: float, Tw: int, Tc: int):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.Tw = Tw
        self.Tc = Tc
        self.t = 0
    
    def gen_lr(self, it: Optional[int] = None):
        self.t += 1
        t = it if it is not None else self.t
        if t < self.Tw:
            return (t / self.Tw) * self.lr_max
        else:
            if t <= self.Tc:
                angle = math.pi * (t - self.Tw) / (self.Tc - self.Tw)
                return self.lr_min + 0.5 * (1. + math.cos(angle)) * (self.lr_max - self.lr_min)  
            else:
                return self.lr_min

