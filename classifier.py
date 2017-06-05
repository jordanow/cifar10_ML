from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA as IPCA
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import keras
import numpy as np


def sklearn_preprocessing(X, y, method='scale'):
    print('Performing sklearn preprocessing via', method)

    if method == 'scale':
        return preprocessing.scale(X)
    elif method == 'standardscaler':
        return StandardScaler().fit_transform(X, y)


def dimensional_reduction(X, y, num_features=200, batch_size=500):
    ipca = IPCA(n_components=num_features, batch_size=batch_size)
    ipca.fit(X, y)
    return ipca.transform(X)


def rf_classifier(X_train, y_train, X_test, y_test, method, estimators,  num_features, preprocessing_method):
    print('Random Forest Classification using estimators',
          estimators, 'and preprocessing via', method)
    classifier = rfc(n_estimators=estimators)

    if method == 'pca':
        print('Performing dimensional reduction with features', num_features)
        X_train = dimensional_reduction(
            X_train.astype(float), y_train, num_features=num_features)
        X_test = dimensional_reduction(
            X_test.astype(float), y_test, num_features=num_features)
    else:
        X_train = sklearn_preprocessing(
            X_train.astype(float), y_train.astype(float), preprocessing_method)
        X_test = sklearn_preprocessing(
            X_test.astype(float), y_test.astype(float), preprocessing_method)

    classifier.fit(X_train, y_train)

    y_test_predicted = classifier.predict(X_test)
    return classifier, X_train, y_train, y_test_predicted


def logistic_classifier(X_train, y_train, X_test, y_test, method, num_features, preprocessing_method):
    print('Logistic Regression with features',
          num_features, 'and preprocessing via', method)
    # By default the 'multi_class' option is set to 'ovr'
    classifier = LogisticRegression()

    if method == 'pca':
        print('Performing dimensional reduction with features', num_features)
        X_train = dimensional_reduction(
            X_train.astype(float), y_train, num_features=num_features)
        X_test = dimensional_reduction(
            X_test.astype(float), y_test, num_features=num_features)
    else:
        # For sklearn we reduce the size of the dataset
        X_train = X_train[:10000]
        y_train = y_train[:10000]

        X_train = sklearn_preprocessing(
            X_train.astype(float), y_train.astype(float), preprocessing_method)
        X_test = sklearn_preprocessing(
            X_test.astype(float), y_test.astype(float), preprocessing_method)

    classifier.fit(X_train, y_train)

    y_test_predicted = classifier.predict(X_test)
    return classifier, X_train, y_train, y_test_predicted


def cnn_classifier(x_train, y_train, x_test, y_test, lr=0.0001, epochs=1, data_augmentation=False):
    print('Performing cnn classification with learning rate =',
          lr, ' and epochs = ', epochs)
    batch_size = 32
    num_classes = 10

    # Data preprocessing
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if data_augmentation == False:
        print('Without data augmentation')
        cnn_model = model.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_test, y_test),
                              shuffle=True)
    else:
        print('Performing data augmentation')
        datagen = ImageDataGenerator(
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        df = datagen.flow(x_train, y_train,
                          batch_size=batch_size)

        cnn_model = model.fit_generator(df,
                                        steps_per_epoch=x_train.shape[0] // batch_size,
                                        epochs=epochs,
                                        validation_data=(x_test, y_test))

    y_test_predicted = model.predict(x_test, batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=32, verbose=1)

    return cnn_model, y_test_predicted, score
