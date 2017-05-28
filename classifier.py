from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA as IPCA


def rfclassifier(X, y, estimators=50):
    print('Random Forest Classification using estimators', estimators)
    classifier = rfc(n_estimators=estimators)
    X_train = preprocessing.scale(X.astype(float))
    y_train = y
    classifier.fit(X_train, y_train)
    return classifier, X_train


def dimensional_reduction(X, y, num_features=200, batch_size=500):
    ipca = IPCA(n_components=num_features, batch_size=batch_size)
    ipca.fit(X, y)
    return ipca.transform(X)


def logistic_classifier(X, y, num_features):
    print('Logistic Classification with features', num_features)
    classifier = LogisticRegression()
    X_train = dimensional_reduction(X.astype(float), y, num_features=num_features)
    y_train = y
    classifier.fit(X_train, y_train)
    return classifier, X_train
