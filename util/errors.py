import math

import numpy as np


def mse(x, y):
    return np.square(np.subtract(x, y)).mean()


def rmse(x, y):
    return math.sqrt(mse(x, y))


def pe(x, y):
    return np.abs(x - y) / (x + 1e-3)


def mpe(x, y):
    return np.mean(pe(x, y))


def max_pe(x, y):
    return pe(x, y).max()