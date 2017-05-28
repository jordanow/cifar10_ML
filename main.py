import seaborn as sn
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import confusion_matrix
import csv
import datetime as dt
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd

# All the data files must go in here
DATA_FILES_FOLDER = 'cifar-10-batches-py'
# csv delimiter to be used
csv_delimiter = ','
# Initialise the X inputs and y outputs
X, y = [], []
# Initialise the X_test and y_test outputs
X_test, y_test = [], []


def get_file_data(fileName):
    fileopen = open(DATA_FILES_FOLDER + '/' + fileName, 'rb')
    dict = pickle.load(fileopen, encoding='bytes')
    fileopen.close()
    return dict


# Read all the data files in DATA_FILES_FOLDER
def get_training_set():
    for filename in os.listdir(DATA_FILES_FOLDER):
        if filename.startswith("data"):
            # print('Reading data from file', filename)
            file_data = get_file_data(filename)
            global X
            if len(X) == 0:
                X = np.array(file_data[b'data'])
            else:
                X = np.append(X, np.array(file_data[b'data']), axis=0)
            global y
            if len(y) == 0:
                y = np.array(file_data[b'labels'])
            else:
                y = np.append(y, np.array(file_data[b'labels']))


# Read the test file in DATA_FILES_FOLDER
def get_test_set():
    for filename in os.listdir(DATA_FILES_FOLDER):
        if filename.startswith("test"):
            # print('Reading data from file', filename)
            file_data = get_file_data(filename)
            global X_test
            if len(X_test) == 0:
                X_test = np.array(file_data[b'data'])
            else:
                X_test = np.append(X_test, np.array(
                    file_data[b'data']), axis=0)
            global y_test
            if len(y_test) == 0:
                y_test = np.array(file_data[b'labels'])
            else:
                y_test = np.append(y_test, np.array(file_data[b'labels']))


def plot_confusion_matrix(cm, labels):
    df_cm = pd.DataFrame(cm, index=[i for i in labels],
                         columns=[i for i in labels])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)


def get_statistics(y_true, y_pred):
    print('accuracy_score', accuracy_score(y_true, y_pred))
    print('precision_score', precision_score(y_true, y_pred, average='weighted'))
    print('recall_score', recall_score(y_true, y_pred, average='weighted'))


print('Started computation at', dt.datetime.now())
get_training_set()
# print(X.shape, y.shape)

classifier = rfc(n_estimators=50)
classifier.fit(X, y)

print("Training set score", classifier.score(X, y))

get_test_set()
# print(X_test.shape, y_true.shape)

y_predicted = classifier.predict(X_test)

print("Test set score", classifier.score(X_test, y_test))
print("Predicted set score", classifier.score(X_test, y_predicted))

# Build confusion matrix
conf_mat = confusion_matrix(y_test, y_predicted)
print(conf_mat)
# plot_confusion_matrix(conf_mat, y_test)

get_statistics(y_test, y_predicted)

print('Ended computation at', dt.datetime.now())
