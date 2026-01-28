import math


def lr_schedule(t: int, lr_max: float, lr_min: float, Tw: int, Tc: int):
    if t < Tw:
        return (t / Tw) * lr_max
    else:
        if t <= Tc:
            angle = math.pi * (t - Tw) / (Tc - Tw)
            return lr_min + 0.5 * (1. + math.cos(angle)) * (lr_max - lr_min)  
        else:
            return lr_min