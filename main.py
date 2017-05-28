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

rfc = classifier.rfclassifier(X, y, 100)
y_predicted = rfc.predict(X_test)

accuracy, precision, recall = st.get_statistics(y_test, y_predicted)
print('Accuracy is :', accuracy)
print('Precision is :', precision)
print('Recall is :', recall)

# print("Test set score", classifier.score(X_test, y_test))
# print("Predicted set score", classifier.score(X_test, y_predicted))

# # Build confusion matrix
# # conf_mat = confusion_matrix(y_test, y_predicted)
# # print(conf_mat)
# # plot_confusion_matrix(conf_mat, y_test)

print('Ended computation at', dt.datetime.now())
