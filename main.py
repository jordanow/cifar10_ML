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

    y_train_predicted = rfc.predict(X_train)
    accuracy, precision, recall, f1_score = st.get_statistics(
        y, y_train_predicted)
    print('Accuracy for training set is :', accuracy)
    print('Precision for training set is :', precision)
    print('Recall for training set is :', recall)
    print('F1 Score for training set is :', f1_score)

    y_test_predicted = rfc.predict(X_test)
    accuracy, precision, recall, f1_score = st.get_statistics(
        y_test, y_test_predicted)
    print('-------------------------------------------')
    print('Accuracy for test set is :', accuracy)
    print('Precision for test set is :', precision)
    print('Recall for test set is :', recall)
    print('F1 Score for test set is :', f1_score)
    print('===========================================')


# Get the statistics for logistic regression classifier
def get_LR_stats(components):
    cl, X_train = classifier.logistic_classifier(X, y, components)

    y_train_predicted = cl.predict(X_train)
    accuracy, precision, recall, f1_score = st.get_statistics(
        y, y_train_predicted)
    print('Accuracy for training set is :', accuracy)
    print('Precision for training set is :', precision)
    print('Recall for training set is :', recall)
    print('F1 Score for training set is :', f1_score)

    X_test_dimensionally_reduced = classifier.dimensional_reduction(
        X_test, y_test, components)
    y_test_predicted = cl.predict(X_test_dimensionally_reduced)
    accuracy, precision, recall, f1_score = st.get_statistics(
        y_test, y_test_predicted)
    print('-------------------------------------------')
    print('Accuracy for test set is :', accuracy)
    print('Precision for test set is :', precision)
    print('Recall for test set is :', recall)
    print('F1 Score for test set is :', f1_score)
    print('===========================================')


# get_LR_stats(300)
# get_rfc_stats(500)

# print("Test set score", classifier.score(X_test, y_test))
# print("Predicted set score", classifier.score(X_test, y_predicted))

# # Build confusion matrix
# # conf_mat = confusion_matrix(y_test, y_predicted)
# # print(conf_mat)
# # plot_confusion_matrix(conf_mat, y_test)


print('Ended computation at', dt.datetime.now())
