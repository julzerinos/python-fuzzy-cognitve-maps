import argparse
import csv
import math

import numpy as np
from tqdm import trange


# step functions

def steps_with_offset(arr, n, offset):
    return [
        {"x": np.array(arr[i:i + n]), "y": np.array(arr[i + n])} for i in range(0, len(arr) - n, offset)
    ]


def distinct_steps(arr, n):
    # return [
    #     {"x": np.array(arr[i:i + n]), "y": np.array(arr[i + n])} for i in range(0, len(arr) + 1 - n, n)
    # ]
    return steps_with_offset(arr, n, n)


def overlap_steps(arr, n):
    # return [
    #     {"x": np.array(arr[i:i + n]), "y": np.array(arr[i + n])} for i in range(0, len(arr) - n)
    # ]
    return steps_with_offset(arr, n, 1)

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
    # n = 3 #no of weights / nodes
    #
    # def func(w, *args):
    #     x = args #values from the time series
    #     result = 0
    #     for i in 1 : n:
    #         result += x[i] * w[i]
    #     return result - x[i+1]
    #
    # lb = -np.ones(n)
    # ub = np.ones(n)
    #
    # fargs = x
    # xopt, fopt = pso(func, lb, ub, args = fargs)
    #
    # return xopt, fopt
    pass


def ga():  # TODO: add genetic algorithm
    # n = 3
    #
    # varbound = np.array([[-1,1]]*n)
    #
    #
    # def f(X):
    #     result = 0
    #     for i in 1 : n:
    #         result+= X[i] * w[i]
    #     return result
    #
    # model=ga(function = f, dimension = n, variable_type = 'real', variable_boundaries = varbound)
    # model.run()
    # convergence = model.report
    # solution = model.output_dict
    # return solution
    pass


#

# model modes (generators)

def inner_calculations(
        time_series,
        fuzzy_nodes, window,
        step, transformation,
        weights, input_weights,
        error, error_max
):
    for step in step(time_series, window):
        yt = transformation(
            np.matmul(
                weights,
                np.einsum("ij,ij->j", input_weights, step['x'])
            )
        )
        e = error(yt, step['y'])

        if error_max < e:
            error_max = e

        weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)  # TODO: weights optimization algorithm
        input_weights = np.random.rand(window, fuzzy_nodes)

    return weights, input_weights, error_max


def outer_calculations(
        time_series,
        fuzzy_nodes, window,
        step, transformation,
        weights, input_weights,
        error, error_max
):
    yts = np.array([])
    ys = np.array([])

    for step in step(time_series, window):
        yt = [transformation(
            np.matmul(
                weights,
                np.einsum("ij,ij->j", input_weights, step['x'])
            )
        )]
        yts = np.concatenate((yts, yt), axis=None)
        ys = np.concatenate(ys, step['y'], axis=None)

    e = error(ys, yts)
    if error_max < e:
        error_max = e

    weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)  # TODO: weights optimization algorithm
    input_weights = np.random.rand(window, fuzzy_nodes)

    return weights, input_weights, error_max


#

# data import

def rescale(min, max):
    return lambda x: (x - min) / (max - min)


def import_and_transform(file, sep=',', header=None):
    with open(file, newline='') as csv_file:
        model_input = np.array(list(csv.reader(csv_file))).astype(np.float)
    max = np.maximum.reduce(model_input)
    min = np.minimum.reduce(model_input)

    return rescale(min, max)(model_input)


def import_from_uwave():
    return import_and_transform("UWaveGestureLibrary/Train/1/10.csv")  # TODO: choose random train/test


#

def main():
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Map Temporal Data Forecaster')
    parser.add_argument('-i', metavar='i',
                        type=int, help='the maximal iteration count',
                        default=500, action='store')
    parser.add_argument('-n', metavar='n',
                        type=int, help='window size',
                        default=4, action='store')
    args = parser.parse_args()

    step = overlap_steps
    transformation = sigmoid()
    error = mse
    mode = inner_calculations

    max_iter = args.i
    performance_index = 0.05
    error_max = -1

    time_series = import_from_uwave()
    fuzzy_nodes = time_series.shape[1]
    window = args.n

    input_weights = np.random.rand(window, fuzzy_nodes)
    weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)

    for _ in trange(max_iter, desc='model iterations', leave=True):
        weights, input_weights, error_max = mode(
            time_series,
            fuzzy_nodes, window,
            step, transformation,
            weights, input_weights,
            error, error_max
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
