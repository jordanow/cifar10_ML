import os
import pickle
import numpy as np


# Read all the data files in given folder matching the teststring
def get_image_data(folder, teststring):
    X, y = [], []
    for filename in os.listdir(folder):
        if filename.startswith(teststring):
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
