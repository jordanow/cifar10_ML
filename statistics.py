from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Return precision, recall, f1-score and support values


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4)


# Returns the accuracy score
def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def plot_confusion_matrix(cm, labels):
    plt.hist(cm, bins=30, normed=True)
    plt.show()


def plot_histogram(y_train):
    # Make histogram of category distribution
    counter = Counter(y_train).most_common()
    classes = [c[0] for c in counter]
    classes_f = [c[1] for c in counter]

    # Plot histogram using matplotlib bar().
    indexes = np.arange(len(classes)) + 0.5
    width = 0.5
    rec = plt.bar(indexes, classes_f, width)
    plt.xticks(indexes + width * 0.5, classes, fontsize=15, rotation=90)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('Classes', fontsize=16)
    plt.title('Frequency Histogram for Classes', fontsize=16)
    plt.tight_layout()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., height + 1,
                     '%d' % int(height),
                     ha='center', va='bottom', fontsize=16)

    autolabel(rec)
    plt.show()


def confusion_matrix(y_test, y_pred):
    list_classes = sorted(list(set(y_test)))
    cm = np.zeros([len(list_classes), len(list_classes)], dtype=int)
    for i in range(len(y_test)):
        cm[list_classes.index(y_test[i]), list_classes.index(y_pred[i])] += 1
    return cm


def plot_confusion_matrix(y_test, y_pred):
    list_classes = sorted(list(set(y_test)))
    cm=confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks=np.arange(len(list_classes))
    plt.xticks(tick_marks, list_classes, rotation=90)
    plt.yticks(tick_marks, list_classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(True)
    width, height=len(list_classes), len(list_classes)
    for x in range(width):
        for y in range(height):
            if cm[x][y] > 100:
                color='white'
            else:
                color='black'
            if cm[x][y] > 0:
                plt.annotate(str(cm[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center',
                             color=color)
    plt.show()
