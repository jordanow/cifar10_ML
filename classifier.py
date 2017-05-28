from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import preprocessing


def rfclassifier(X, y, estimators=50):
    print('Classification using estimators', estimators)
    classifier = rfc(n_estimators=estimators)
    X_train = preprocessing.scale(X.astype(float))
    y_train = y
    classifier.fit(X, y)
    return classifier
