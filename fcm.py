import argparse
import csv
import math
import os
import random
import signal
import sys
import time
from contextlib import contextmanager

import numpy as np
import scipy.optimize as optimize
from lmfit import Parameters, Minimizer
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


LAST_SIGNAL = 0


def signal_handle(sig, frame):
    global LAST_SIGNAL
    LAST_SIGNAL = sig


# step functions

def steps_with_offset(arr, n, offset):
    return [
        {"x": np.array(arr[i:i + n]), "y": np.array(arr[i + n])} for i in range(0, len(arr) - n, offset)
    ]


def distinct_steps(arr, n):
    return steps_with_offset(arr, n, n)


def overlap_steps(arr, n):
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


def pe(x, y):
    return np.abs((x - y) / (x + 1e-7))


def mpe(x, y):
    return np.mean(pe(x, y))


def max_pe(x, y):
    return pe(x, y).max()


#

# weight optimization algorithms


def scipy(fcm_weights, agg_weights, const, func):
    flat_weights = np.concatenate((fcm_weights.flatten(), agg_weights.flatten()), axis=None)

    bounds = optimize.Bounds(-np.ones(flat_weights.shape), np.ones(flat_weights.shape))
    nonlinc = optimize.NonlinearConstraint(func, 0, 0)

    res = optimize.minimize(func, flat_weights, method='trust-constr', bounds=bounds,
                            # constraints=nonlinc,
                            options={'disp': True, 'maxiter': 300, 'xtol': 1e-10})
    # res = optimize.minimize(func, flat_weights, method='trust-constr', constraints=nonlinc, options={'disp': True}, bounds=bnds)

    n, m = const

    err = func(flat_weights)

    fcm_weights = np.reshape(res.x[:n * n], (n, n))
    agg_weights = np.reshape(res.x[n * n:], (m, n))

    return fcm_weights, agg_weights, err


def scipy_inner(
        transformation,
        fcm_weights, agg_weights,
        x, y,
        error
):
    n = fcm_weights.shape[0]
    m = agg_weights.shape[0]

    def func(w):
        fw = np.reshape(w[:n * n], (n, n))
        aw = np.reshape(w[n * n:], (m, n))

        yt = calc(transformation, fw, aw, x)

        return error(yt, y)

    const = n, m

    fcm_weights, agg_weights, err = scipy(fcm_weights, agg_weights, const, func)

    return fcm_weights, agg_weights, err


def scipy_outer(
        transformation,
        fcm_weights, agg_weights,
        time_series, step, window,
        error
):
    n = fcm_weights.shape[0]
    m = agg_weights.shape[0]

    def func(w):
        fw = np.reshape(w[:n * n], (n, n))
        aw = np.reshape(w[n * n:], (m, n))

        yts, ys = calc_all(time_series, step, window, transformation, fw, aw)

        err = error(ys, yts)

        return err

    const = n, m

    fcm_weights, agg_weights, e = scipy(fcm_weights, agg_weights, const, func)

    return fcm_weights, agg_weights, e


def lmfit(fcm_weights, agg_weights, const, func):
    flat_weights = np.concatenate((fcm_weights.flatten(), agg_weights.flatten()), axis=None)

    params = Parameters()

    np.fromiter(map(lambda x: params.add(f'w{x[0]}', value=x[1], min=-1, max=1), enumerate(flat_weights)), dtype=float)

    fitter = Minimizer(func, params)
    result = fitter.minimize(method='nelder')

    n, m = const

    err = func(result.params)

    fcm_weights = np.reshape(np.fromiter([result.params[f'w{i}'] for i in range(n * n)], dtype=float), (n, n))
    agg_weights = np.reshape(
        np.fromiter([result.params[f'w{i}'] for i in range(n * n, len(flat_weights))], dtype=float), (m, n))

    return fcm_weights, agg_weights, err


def lmfit_inner(
        transformation,
        fcm_weights, agg_weights,
        x, y,
        error
):
    n = fcm_weights.shape[0]
    m = agg_weights.shape[0]

    def func(w):
        fw = np.reshape(np.fromiter([w[f'w{i}'] for i in range(n * n)], dtype=float), (n, n))
        aw = np.reshape(np.fromiter([w[f'w{i}'] for i in range(n * n, len(w))], dtype=float), (m, n))

        yt = calc(transformation, fw, aw, x)

        return error(y, yt)

    const = n, m

    fcm_weights, agg_weights, err = lmfit(fcm_weights, agg_weights, const, func)

    return fcm_weights, agg_weights, err


def lmfit_outer(
        transformation,
        fcm_weights, agg_weights,
        time_series, step, window,
        error
):
    n = fcm_weights.shape[0]
    m = agg_weights.shape[0]

    def func(w):
        fw = np.reshape(np.fromiter([w[f'w{i}'] for i in range(n * n)], dtype=float), (n, n))
        aw = np.reshape(np.fromiter([w[f'w{i}'] for i in range(n * n, len(w))], dtype=float), (m, n))

        yts, ys = calc_all(time_series, step, window, transformation, fw, aw)

        err = error(ys, yts)

        return err

    const = n, m

    fcm_weights, agg_weights, e = lmfit(fcm_weights, agg_weights, const, func)

    return fcm_weights, agg_weights, e


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
        weights, input_weights, e = lmfit_inner(
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
    weights, input_weights, e = lmfit_outer(
        transformation,
        weights, input_weights,
        time_series, step, window,
        error
    )

    return weights, input_weights, e


#


# data import

def rescale(min, max):
    return lambda x: np.subtract(x, min) / np.subtract(max, min)


def import_and_transform(file_train, file_test, sep=',', header=None):
    with open(file_train, newline='') as csv_file:
        model_input_train = np.array(list(csv.reader(csv_file))).astype(np.float)
    with open(file_test, newline='') as csv_file:
        model_input_test = np.array(list(csv.reader(csv_file))).astype(np.float)

    model_input = np.concatenate((model_input_train, model_input_test))

    max = model_input.max(0)
    min = model_input.min(0)

    return rescale(min, max)(model_input_train), rescale(min, max)(model_input_test)


def import_from_uwave(c=1, train=-1, test=-1):
    train_path = 'UWaveGestureLibrary/Train'
    train_files = os.listdir(f'{train_path}/{c}')
    train_index = random.randrange(0, len(train_files))
    train_file = f'{train_path}/{c}/{train_files[train_index]}'

    test_path = 'UWaveGestureLibrary/Test'
    test_files = os.listdir(f'{test_path}/{c}')
    test_index = random.randrange(0, len(test_files))
    test_file = f'{test_path}/{c}/{test_files[test_index]}'

    train_series, test_series = import_and_transform(train_file, test_file)

    return train_series, test_series, train_file, test_file


#

def main():
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Map Temporal Data Forecaster')
    parser.add_argument('-i', metavar='i',
                        type=int, help='the maximal iteration count',
                        default=250, action='store')
    parser.add_argument('-n', metavar='n',
                        type=int, help='window size',
                        default=4, action='store')
    args = parser.parse_args()

    global LAST_SIGNAL
    signal.signal(signal.SIGINT, signal_handle)

    step = overlap_steps

    transformation = sigmoid
    error = max_pe
    mode = outer_calculations

    max_iter = args.i
    performance_index = 1e-5

    errors = []
    loop_error = 0

    data_class = 1
    train_series, test_series, train_file, test_file = import_from_uwave(data_class)
    fuzzy_nodes = train_series.shape[1]
    window = args.n

    input_weights = np.random.rand(window, fuzzy_nodes)
    weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)

    for _ in trange(max_iter, desc='model iterations', leave=True):
        weights, input_weights, loop_error = mode(
            train_series,
            fuzzy_nodes, window,
            step, transformation(),
            weights, input_weights,
            error
        )

        print("loop_error: ", loop_error)

        errors.append(loop_error)

        if loop_error <= performance_index or LAST_SIGNAL == signal.SIGINT:
            break

    ts = int(time.time())

    if not os.path.exists('output'):
        os.makedirs('output')

    f = open(f"output/{ts}.txt", "a")

    f.write("setup details ---------------\n")
    f.write(f"train file {train_file}\n")
    f.write(f"test file {test_file}\n")
    f.write(f"weights optimizer: TBA\n")
    f.write(f"error function: {error.__name__}\n")
    f.write(f"step function: {step.__name__}\n")
    f.write(f"transformation function: {transformation.__name__}\n")
    f.write(f"calculations position: {mode.__name__}\n")
    f.write(f"max iters: {max_iter}\n")
    f.write(f"window: {str(window)}\n")
    f.write("\n")
    f.write("optimized weights matrix\n")
    f.write(f"{str(weights)}\n")
    f.write("\n")
    f.write("final error ? performance index\n")
    f.write(f"{loop_error} {'>' if loop_error > performance_index else '<='} {performance_index}\n")
    f.write("\n")
    f.write("iterations\n")
    f.write(f"{str(len(errors))}\n")

    f1 = plt.figure(1)
    f1.suptitle('Train errors')
    plt.ylabel(f'{error.__name__}')
    plt.xlabel('outer loop iteration count')
    plt.plot(errors)
    plt.savefig(f'output/train_errors_{ts}.png', bbox_inches='tight')

    test_errors = []
    for step in step(test_series, window):
        yt = calc(transformation(), weights, input_weights, step['x'])

        test_errors.append(error(yt, step['y']))

    f2 = plt.figure(2)
    f2.suptitle('Test errors')
    plt.ylabel(f'{error.__name__}')
    plt.xlabel('nth forecast vs target')
    plt.plot(test_errors)

    plt.savefig(f'output/test_errors_{ts}.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
