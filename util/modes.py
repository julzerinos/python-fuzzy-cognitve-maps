import numpy as np
import scipy.optimize as optimize
from lmfit import Parameters, Minimizer


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

        return error(y, yt)

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
