import datetime as dt
from keras.datasets import cifar10
import numpy as np
from sklearn import preprocessing

import file_read as fr
import statistics as st
import classifier

# All the data files must go in here
DATA_FILES_FOLDER = 'cifar-10-batches-py'


print('Started computation at', dt.datetime.now())

# Fetch the training data
X, y = fr.get_image_data(DATA_FILES_FOLDER, 'data')
X_test, y_test = fr.get_image_data(DATA_FILES_FOLDER, 'test')
# print(X.shape, y.shape)
# print(X_test.shape, y_test.shape)


# Get the statistics for Random forest classifier
def get_rfc_stats(method='sklearn', estimators=50, num_features=200, preprocessing_method='scale'):
    cl_model, X_train, y_train, y_test_predicted = classifier.rf_classifier(
        X, y, X_test, y_test, method, estimators,  num_features, preprocessing_method)

    st.n_fold_cross_validation_score(cl_model, X_train, y_train, 10)

    print(st.get_classification_report(
        y_test, y_test_predicted))

    print('Accuracy of test :', st.get_accuracy(
        y_test, y_test_predicted))

    st.plot_confusion_matrix(y_test, y_test_predicted)

    print('===========================================')


# Get the statistics for logistic regression classifier
def get_LR_stats(method='sklearn', num_features=200, preprocessing_method='scale'):
    cl_model, X_train, y_train, y_test_predicted = classifier.logistic_classifier(
        X, y, X_test, y_test, method,  num_features, preprocessing_method)

    st.n_fold_cross_validation_score(cl_model, X_train, y_train, 10)

    print(st.get_classification_report(
        y_test, y_test_predicted))

    print('Accuracy of test :', st.get_accuracy(
        y_test, y_test_predicted))

    st.plot_confusion_matrix(y_test, y_test_predicted)

    print('===========================================')


def get_CNN_stats(lr, epochs, data_augmentation):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    model, y_test_predicted, score = classifier.cnn_classifier(
        x_train, y_train, x_test, y_test, lr, epochs, data_augmentation)

    y_test_predicted = np.argmax(y_test_predicted, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(st.get_classification_report(
        y_test, y_test_predicted))

    print('Accuracy of test :', st.get_accuracy(
        y_test, y_test_predicted))

    print('Score of test', score[1])

    print('time until accuracy and everythng', dt.datetime.now())

    # plot model history
    st.plot_cnn_stats(model)

    print('===========================================')


# st.plot_histogram(y)
# st.plot_histogram(y_test)

# With sklearn preprocessing
get_rfc_stats(method='pca', estimators=400,
              num_features=200, preprocessing_method='scale')
get_LR_stats(method='sklearn', num_features='', preprocessing_method='scale')

# With PCA
get_rfc_stats(method='pca', estimators=400, num_features=200, preprocessing_method='scale')
get_LR_stats(method='pca', num_features=200, preprocessing_method='scale')

# CNN
get_CNN_stats(lr=0.001, epochs=3, data_augmentation=True)

print('Ended computation at', dt.datetime.now())
