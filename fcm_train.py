import json
import os
import random
import signal
import sys
import time
import argparse
from contextlib import contextmanager

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

from util import modes, errors as err, data, transformations as trans, steps


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


def main(args):
    global LAST_SIGNAL
    signal.signal(signal.SIGINT, signal_handle)

    print(args)
    return

    steparg, transform, errorarg, modearg, iterarg, pi = args

    if (steparg == "overlap"):
        step = steps.overlap_steps
    else:
        step = steps.distinct_steps

    if (transform == "sigmoid"):
        transformation = trans.sigmoid
    elif (transform == "binary"):
        transformation = trans.binary
    elif (transform == "tanh"):
        transformation = trans.tanh
    elif (transform == "arctan"):
        transformation = trans.arctan
    else:
        transformation = trans.gaussian

    if (errorarg == "rmse"):
        error = err.rmse
    elif (errorarg == "mpe"):
        error = err.mpe
    else:
        error = err.max_pe

    if (modearg == "outer"):
        mode = modes.outer_calculations
    else:
        mode = modes.inner_calculations

    max_iter = iterarg
    performance_index = pi

    errors = []
    loop_error = 0

    amount = 4
    train_path = 'UWaveGestureLibrary/Train'
    test_path = 'UWaveGestureLibrary/Test'
    classif = 1
    train_series_set, test_series, train_file, test_file = data.import_from_uwave(amount, train_path=train_path,
                                                                                  test_path=test_path, classif=classif)
    fuzzy_nodes = train_series_set[0].shape[1]
    window = 4

    input_weights = np.random.rand(window, fuzzy_nodes)
    weights = np.random.rand(fuzzy_nodes, fuzzy_nodes)

    for _ in trange(max_iter, desc='model iterations', leave=True):
        weights, input_weights, loop_error = mode(
            random.choice(train_series_set),
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

    if not os.path.exists(f'output/{ts}'):
        os.makedirs(f'output/{ts}')

    summary = {
        'config': {
            'step': step.__name__,
            'algorithm': 'Nelder-Mead',
            'error': error.__name__,
            'transformation function': transformation.__name__,
            'calculations position': mode.__name__,
            'max iterations': max_iter,
            'window size': window,
            'performance index': performance_index
        },
        'files': {
            'training': train_file,
            'testing': test_file[0],
            'train path': train_path,
            'test path': test_path,
            'class': classif
        },
        'weights': {
            'aggregation': input_weights.tolist(),
            'fcm': weights.tolist()
        },
        'results': {
            'final error': loop_error,
            'iterations': len(errors),
            'errors': errors
        }
    }

    with open(f"output/{ts}/train_summary.json", "w") as f:
        json.dump(summary, f)

    f1 = plt.figure(1)
    f1.suptitle('Train errors')
    plt.ylabel(f'{error.__name__}')
    plt.xlabel('outer loop iteration count')
    plt.plot(errors)
    plt.savefig(f'output/{ts}/train_errors.png', bbox_inches='tight')

    return ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Mapping training')
    parser.add_argument('-s', '--step', dest='step', default="overlap", choices=["overlap", "distinct"], type=str, help='Steps for training')
    parser.add_argument('-t', '--transformation', dest='transform', default="sigmoid",
     choices=["sigmoid", "binary", "tanh", "arctan", "gaussian"],
     type=str, help='Transformation function')
    parser.add_argument('-e', '--error', dest='error', default="rmse",
     choices=["rmse", "mpe", "max_pe"], type=str, help='Error function')  
    parser.add_argument('-m', '--mode', dest='mode', default="outer",
     choices=["outer", "inner"], type=str, help='Mode of calculations')
    parser.add_argument("-i", "--iter", dest="iter", default=500, type=int, help='Training iterations')
    parser.add_argument("-p", "--performance", dest="pi", default=1e-5, type=float, help='Performance index')

    args = parser.parse_args()
    argu = args.step, args.transform, args.error, args.mode, args.iter, args.pi
    main(argu)
