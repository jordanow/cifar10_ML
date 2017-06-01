import datetime as dt

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
def get_rfc_stats(estimators, method, num_features):
    rfc, X_train = classifier.rfclassifier(
        X, y, estimators, method, num_features)

    st.n_fold_cross_validation(rfc, X_train, y, 10)

    # We need to do this as the classifier only expect num_features in
    # the input data
    if method == 'dimensional_reduction':
        global X_test
        X_test = classifier.dimensional_reduction(X_test, y_test, num_features=num_features)

    y_test_predicted = rfc.predict(X_test)
    print(st.get_classification_report(
        y_test, y_test_predicted))

    print('Accuracy of test :', st.get_accuracy(
        y_test, y_test_predicted))

    st.plot_confusion_matrix(y_test, y_test_predicted)

    print('===========================================')


# Get the statistics for logistic regression classifier
def get_LR_stats(num_features):
    cl, X_train = classifier.logistic_classifier(X, y, num_features)

    st.n_fold_cross_validation(cl, X_train, y, 10)

    # We need to do this as the classifier only expect num_features in
    # the input data
    X_test = classifier.dimensional_reduction(X_test, y_test, num_features=num_features)

    y_test_predicted = cl.predict(X_test)
    print(st.get_classification_report(
        y_test, y_test_predicted))

    print('Accuracy of test :', st.get_accuracy(
        y_test, y_test_predicted))

    st.plot_confusion_matrix(y_test, y_test_predicted)

    print('===========================================')


# st.plot_histogram(y)
# st.plot_histogram(y_test)
# get_LR_stats(300)

# With sklearn preprocessing
# get_rfc_stats(5)

# With Dimensional reduction
get_rfc_stats(50, method='dimensional_reduction', num_features=50)


# print("Test set score", classifier.score(X_test, y_test))
# print("Predicted set score", classifier.score(X_test, y_predicted))

# # Build confusion matrix
# # conf_mat = confusion_matrix(y_test, y_predicted)
# # print(conf_mat)
# # plot_confusion_matrix(conf_mat, y_test)


print('Ended computation at', dt.datetime.now())
