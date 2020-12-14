import math

import numpy as np


def sigmoid(t=5):
    return lambda x: 1 / (1 + np.exp(t * -x))


def binary():
    return lambda x: 0 if x < 0 else 1


def tanh():
    return lambda x: np.tanh(x)


def arctan():
    return lambda x: np.arctan(x)


def gaussian():
    return lambda x: math.e ** ((-x) ** 2)
