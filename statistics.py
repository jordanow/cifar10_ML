from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Return accuracy, precision and recall scores
def get_statistics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average='weighted'), recall_score(y_true, y_pred, average='weighted')


def plot_confusion_matrix(cm, labels):
    plt.hist(cm, bins=30, normed=True)
    plt.show()
