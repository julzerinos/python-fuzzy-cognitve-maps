import argparse
import math

import numpy as np
from tqdm import trange


# step functions

def distinct_steps(arr, n):
    return [
        np.array(arr[i:i + n]) for i in range(0, len(arr) + 1 - n, n)
    ]


def overlap_steps(arr, n):
    return [
        np.array(arr[i:i + n]) for i in range(0, len(arr) + 1 - n)
    ]


#


# transformation functions

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


#

# error functions

def mse(x, y):
    return np.square(np.subtract(x, y)).mean()


def rmse(x, y):
    return math.sqrt(mse(x, y))


#

# weight optimization algorithms

def pso():  # TODO: add particle swarm optimization
    pass


def ga():  # TODO: add genetic algorithm
    pass


#

# model modes (generators)

def inner_calculations(time_series, fuzzy_nodes, step, transformation, error, weights, error_max):
    for x in step(time_series, fuzzy_nodes):
        y = transformation(np.matmul(weights, x))

        e = error(x, y)
        if error_max < e:
            error_max = e

        weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)  # TODO: weights optimization algorithm

    return weights, error_max


def outer_calculations(time_series, fuzzy_nodes, step, transformation, error, weights, error_max):
    y_all = np.array([np.zeros(fuzzy_nodes - 1)])
    # TODO: understand step function
    #       for outer calculations, always take distinct forecast sets?
    for x in distinct_steps(time_series, fuzzy_nodes):
        y = [transformation(np.matmul(weights, x))]
        y_all = np.concatenate((y_all, y), axis=None)

    e = error(time_series, y_all)
    if error_max < e:
        error_max = e

    weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)  # TODO: weights optimization algorithm

    return weights, error_max


#

def main():
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Map Temporal Data Forecaster')
    parser.add_argument('-n', metavar='n',
                        type=int, help='the number of fuzzy nodes',
                        default=3, action='store')
    parser.add_argument('-i', metavar='i',
                        type=int, help='the maximal iteration count',
                        default=500, action='store')
    args = parser.parse_args()

    fuzzy_nodes = args.n
    time_series = np.random.rand(500)  # TODO: change to file read or from functions

    max_iter = args.i

    performance_index = 0.05
    error_max = -1

    weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)

    step = overlap_steps
    transformation = sigmoid()
    error = mse
    mode = inner_calculations

    for _ in trange(max_iter, desc='model iterations', leave=True):
        weights, error_max = mode(
            time_series, fuzzy_nodes,
            step, transformation, error,
            weights, error_max
        )

        if error_max < performance_index:
            break

    # TODO: display results

    print()
    print("optimized weights matrix")
    print(weights)
    print()
    print("final error ? performance index")
    print(error_max, '>' if error_max > performance_index else '<=', performance_index)

    # TODO: error graph (matplotlib)
    # TODO: source vs forecast values graph


if __name__ == '__main__':
    main()
