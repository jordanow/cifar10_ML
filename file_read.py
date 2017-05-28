import os
import pickle
import numpy as np

# Read all the data files in given folder


def get_training_set(folder):
    X, y = [], []
    for filename in os.listdir(folder):
        if filename.startswith("data"):
            # print('Reading data from file', filename)
            file_data = get_file_data(folder + '/' + filename)
            if len(X) == 0:
                X = np.array(file_data[b'data'])
            else:
                X = np.append(X, np.array(file_data[b'data']), axis=0)
            if len(y) == 0:
                y = np.array(file_data[b'labels'])
            else:
                y = np.append(y, np.array(file_data[b'labels']))
    return X, y


def get_file_data(fileName):
    fileopen = open(fileName, 'rb')
    dict = pickle.load(fileopen, encoding='bytes')
    fileopen.close()
    return dict


# Read the test file in given folder
def get_test_set(folder):
    X_test, y_test = [], []
    for filename in os.listdir(folder):
        if filename.startswith("test"):
            # print('Reading data from file', filename)
            file_data = get_file_data(folder + '/' + filename)
            if len(X_test) == 0:
                X_test = np.array(file_data[b'data'])
            else:
                X_test = np.append(X_test, np.array(
                    file_data[b'data']), axis=0)
            if len(y_test) == 0:
                y_test = np.array(file_data[b'labels'])
            else:
                y_test = np.append(y_test, np.array(file_data[b'labels']))
    return X_test, y_test
