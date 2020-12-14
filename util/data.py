import csv
import os
import random

import numpy as np


def rescale(min, max):
    return lambda x: np.subtract(x, min) / np.subtract(max, min)


def import_and_transform(train_files, test_file, train_path, test_path, classif, sep=',', header=None):
    model_input_train = []
    for t in train_files:
        with open(f'{train_path}/{classif}/{t}', newline='') as csv_file:
            model_input_train.append(np.array(list(csv.reader(csv_file))).astype(np.float))
    with open(f'{test_path}/{classif}/{test_file}', newline='') as csv_file:
        model_input_test = np.array(list(csv.reader(csv_file))).astype(np.float)

    model_input = np.concatenate((model_input_test, model_input_train[0]))
    for i, t in enumerate(model_input_train):
        if i == 0:
            continue
        model_input = np.concatenate((model_input, t))

    max = model_input.max(0)
    min = model_input.min(0)

    for t in model_input_train:
        model_input_train = rescale(min, max)(model_input_train)

    return model_input_train, rescale(min, max)(model_input_test)


def import_from_uwave(amount=1, train_path='UWaveGestureLibrary/Train', test_path='UWaveGestureLibrary/Test',
                      classif=1):
    train_files = random.sample(os.listdir(f'{train_path}/{classif}'), amount)
    test_files = random.sample(os.listdir(f'{test_path}/{classif}'), 1)

    train_series_set, test_series = import_and_transform(train_files, test_files[0], train_path, test_path, classif)

    return train_series_set, test_series, train_files, test_files
