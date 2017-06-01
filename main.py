import datetime as dt

import file_read as fr
import statistics as st
import classifier

# All the data files must go in here
DATA_FILES_FOLDER = 'cifar-10-batches-py'


print('Started computation at', dt.datetime.now())

X, y = fr.get_image_data(DATA_FILES_FOLDER, 'data')
X_test, y_test = fr.get_image_data(DATA_FILES_FOLDER, 'test')
# print(X.shape, y.shape)
# print(X_test.shape, y_test.shape)


# Get the statistics for Random forest classifier
def get_rfc_stats(estimators):
    rfc, X_train = classifier.rfclassifier(X, y, estimators)

    st.n_fold_cross_validation(rfc, X_train, y, 20)

    y_test_predicted = rfc.predict(X_test)
    print(st.get_classification_report(
        y_test, y_test_predicted))

    print('Accuracy of test :', st.get_accuracy(
        y_test, y_test_predicted))

    st.plot_confusion_matrix(y_test, y_test_predicted)

    print('===========================================')


# Get the statistics for logistic regression classifier
def get_LR_stats(components):
    cl, X_train = classifier.logistic_classifier(X, y, components)

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
get_rfc_stats(1)

# print("Test set score", classifier.score(X_test, y_test))
# print("Predicted set score", classifier.score(X_test, y_predicted))

# # Build confusion matrix
# # conf_mat = confusion_matrix(y_test, y_predicted)
# # print(conf_mat)
# # plot_confusion_matrix(conf_mat, y_test)


print('Ended computation at', dt.datetime.now())
