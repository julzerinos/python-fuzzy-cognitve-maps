import json
import argparse
from os import path

import numpy as np
from matplotlib import pyplot as plt

from util import data, transformations, modes, steps, errors


def main(ts, test_file=None):
    if not path.exists(f"output/{ts}/train_summary.json"):
        print("Output folder not found or no test id given")
        return

    with open(f"output/{ts}/train_summary.json", "r") as f:
        train_summary = json.load(f)

    f = open(f"output/{ts}/test_summary.txt", "a")

    f.write("\n")
    f.write(f"i = 0 : test series {train_summary['files']['testing']}\n")
    for i, fn in enumerate(train_summary['files']['training']):
        f.write(f"i = {i + 1} : train series {fn}\n")

    train_series_set, test_series = data.import_and_transform(
        train_summary['files']['training'],
        train_summary['files']['testing'] if test_file is None else test_file,
        train_summary["files"]["train path"],
        train_summary['files']['test path'],
        train_summary["files"]["class"]
    )

    series = [test_series]
    series.extend(train_series_set)

    for i, series in enumerate(series):
        test_errors = {'rmse': [], 'mpe': [], 'max_pe': []}
        for step_i in getattr(steps, train_summary['config']['step'])(series, train_summary['config']['window size']):
            yt = modes.calc(getattr(transformations, train_summary['config']['transformation function'])(),
                            np.array(train_summary['weights']['fcm']),
                            np.array(train_summary['weights']['aggregation']), step_i['x'])

            test_errors['rmse'].append(errors.rmse(step_i['y'], yt))
            test_errors['mpe'].append(errors.mpe(step_i['y'], yt))
            test_errors['max_pe'].append(errors.max_pe(step_i['y'], yt))

        for j, err in enumerate(test_errors):
            f2 = plt.figure(j + 2)
            f2.suptitle(f'{err} errors')
            plt.ylabel(f'{err}')
            plt.xlabel('nth forecast vs target')
            plt.plot(test_errors[err])

            plt.savefig(f'output/{ts}/{i}_{err}_errors.png', bbox_inches='tight')

        f.write("\n")
        f.write(f"{i} test errors\n")
        f.write(
            f"rmse max {np.array(test_errors['rmse']).max()} min {np.array(test_errors['rmse']).min()} final {test_errors['rmse'][-1]}\n")
        f.write(
            f"mpe max {np.array(test_errors['mpe']).max()} min {np.array(test_errors['mpe']).min()} final {test_errors['mpe'][-1]}\n")
        f.write(
            f"max_pe max {np.array(test_errors['max_pe']).max()} min {np.array(test_errors['max_pe']).min()} final {test_errors['max_pe'][-1]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Cognitive Mapping testing')
    parser.add_argument('-ts', dest='ts', default='1607964137', type=str, help='Path to the folder with the weights')
    parser.add_argument('-tf', dest='tf', default=None, type=str, help='test file name (eg. "356.csv"')

    args = parser.parse_args()

    main(args.ts, args.tf)
