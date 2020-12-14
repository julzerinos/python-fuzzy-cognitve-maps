import numpy as np


def steps_with_offset(arr, n, offset):
    return [
        {"x": np.array(arr[i:i + n]), "y": np.array(arr[i + n])} for i in range(0, len(arr) - n, offset)
    ]


def distinct_steps(arr, n):
    return steps_with_offset(arr, n, n)


def overlap_steps(arr, n):
    return steps_with_offset(arr, n, 1)
