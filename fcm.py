import argparse
import csv
import math
import os
import sys
from contextlib import contextmanager

import numpy as np
import pyswarm
from matplotlib import pyplot as plt
from tqdm import trange


# helper

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


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

# TODO: Average of all variable errors


def mse(x, y):
    return np.square(np.subtract(x, y)).mean()


def rmse(x, y):
    return math.sqrt(mse(x, y))


# TODO: Percent error


#

# weight optimization algorithms

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


def pso(fcm_weights, agg_weights, bounds_range, const, func):
    n, m = const

    bounds = np.concatenate((fcm_weights.flatten(), agg_weights.flatten()), axis=None)
    lb = np.vectorize(lambda t: min(t - bounds_range, 1))(bounds)
    ub = np.vectorize(lambda t: max(t + bounds_range, -1))(bounds)

    with suppress_stdout():
        xopt, fopt = pyswarm.pso(func, lb, ub, args=const, debug=False)

    fcm_weights = np.reshape(xopt[:n * n], (n, n))
    agg_weights = np.reshape(xopt[n * n:], (m, n))

    return fcm_weights, agg_weights, fopt


def pso_inner(
        transformation,
        fcm_weights, agg_weights,
        x, y,
        error,
        bounds_range=.25
):
    def func(w, *args):
        xi, yi, ni, mi = args

        fw = np.reshape(w[:ni * ni], (ni, ni))
        aw = np.reshape(w[ni * ni:], (mi, ni))

        yt = calc(transformation, fw, aw, xi)
        e = error(yt, yi)

        return e

    n = fcm_weights.shape[0]
    m = agg_weights.shape[0]
    const = (n, m, x, y)

    fcm_weights, agg_weights, fopt = pso(fcm_weights, agg_weights, bounds_range, const, func)

    return fcm_weights, agg_weights, fopt


def pso_outer(
        transformation,
        fcm_weights, agg_weights,
        time_series, step, window,
        error,
        bounds_range=.25
):
    def func(w, *args):
        ni, mi = args

        fw = np.reshape(w[:ni * ni], (n, n))
        aw = np.reshape(w[ni * ni:], (m, n))

        yts, ys = calc_all(time_series, step, window, transformation, fw, aw)
        e = error(yts, ys)

        return e

    n = fcm_weights.shape[0]
    m = agg_weights.shape[0]
    const = (n, m)

    fcm_weights, agg_weights, fopt = pso(fcm_weights, agg_weights, bounds_range, const, func)

    return fcm_weights, agg_weights, fopt


#

# model modes

def calc(transformation, weights, input_weights, x):
    return transformation(
        np.matmul(
            weights,
            np.einsum("ij,ij->j", input_weights, x)
        )
    )


def calc_all(time_series, step, window, transformation, weights, input_weights):
    yts = np.array([])
    ys = np.array([])

    for step in step(time_series, window):
        yt = [calc(transformation, weights, input_weights, step['x'])]
        yts = np.concatenate((yts, yt), axis=None)
        ys = np.concatenate((ys, step['y']), axis=None)

    return yts, ys


def inner_calculations(
        time_series,
        fuzzy_nodes, window,
        step, transformation,
        weights, input_weights,
        error
):
    error_max = -1

    for step in step(time_series, window):
        weights, input_weights, e = pso_inner(
            transformation,
            weights, input_weights,
            step['x'], step['y'],
            error
        )

        if error_max < e:
            error_max = e

    return weights, input_weights, error_max


def outer_calculations(
        time_series,
        fuzzy_nodes, window,
        step, transformation,
        weights, input_weights,
        error
):
    weights, input_weights, e = pso_outer(
        transformation,
        weights, input_weights,
        time_series, step, window,
        error
    )

    return weights, input_weights, e


#

# TODO: Add feature classification weights & voting during loops


# data import

# TODO: Rescale within variable scope
def rescale(min, max):
    return lambda x: (x - min) / (max - min)


# TODO: Transform test and train
def import_and_transform(file, sep=',', header=None):
    with open(file, newline='') as csv_file:
        model_input = np.array(list(csv.reader(csv_file))).astype(np.float)
    max = np.maximum.reduce(model_input)
    min = np.minimum.reduce(model_input)

    return rescale(min, max)(model_input)


# TODO: Fuzzy c-means for scalar time series
#   - Fuzzy clustering
#   - Python framework
# TODO: classes
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
    error = rmse
    mode = outer_calculations

    max_iter = args.i
    performance_index = 0.01

    errors = []
    loop_error = 0

    time_series = import_from_uwave()
    fuzzy_nodes = time_series.shape[1]
    window = args.n

    input_weights = np.random.rand(window, fuzzy_nodes)
    weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)

    for _ in trange(max_iter, desc='model iterations', leave=True):
        weights, input_weights, loop_error = mode(
            time_series,
            fuzzy_nodes, window,
            step, transformation,
            weights, input_weights,
            error
        )

        errors.append(loop_error)

        if loop_error <= performance_index:
            break

    # TODO: display results

    print()
    print("optimized weights matrix")
    print(weights)
    print()
    print("final error ? performance index")
    print(loop_error, '>' if loop_error > performance_index else '<=', performance_index)
    print()
    print("iterations")
    print(len(errors))

    plt.plot(errors)
    plt.show()

    # TODO: source vs forecast values graph


if __name__ == '__main__':
    main()
