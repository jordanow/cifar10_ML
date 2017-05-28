from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import confusion_matrix
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score

import file_read as fr

# All the data files must go in here
DATA_FILES_FOLDER = 'cifar-10-batches-py'
# csv delimiter to be used
csv_delimiter = ','
# Initialise the X inputs and y outputs
X, y = [], []
# Initialise the X_test and y_test outputs
X_test, y_test = [], []


def plot_confusion_matrix(cm, labels):
    plt.hist(cm, bins=30, normed=True)
    plt.show()


def get_statistics(y_true, y_pred):
    print('accuracy_score', accuracy_score(y_true, y_pred))
    print('precision_score', precision_score(
        y_true, y_pred, average='weighted'))
    print('recall_score', recall_score(y_true, y_pred, average='weighted'))


print('Started computation at', dt.datetime.now())
X, y = fr.get_training_set(DATA_FILES_FOLDER)
X_test, y_test = fr.get_test_set(DATA_FILES_FOLDER)
# print(X_test.shape, y_test.shape)
# print(X.shape, y.shape)

# classifier = rfc(n_estimators=50)

# X_train = preprocessing.scale(X)
# y_train = y
# classifier.fit(X, y)

# print("Training set score", classifier.score(X, y))

# y_predicted = classifier.predict(X_test)

# print("Test set score", classifier.score(X_test, y_test))
# print("Predicted set score", classifier.score(X_test, y_predicted))

# # Build confusion matrix
# # conf_mat = confusion_matrix(y_test, y_predicted)
# # print(conf_mat)
# # plot_confusion_matrix(conf_mat, y_test)

# get_statistics(y_test, y_predicted)

print('Ended computation at', dt.datetime.now())
