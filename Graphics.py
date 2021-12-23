import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def boxplot(df:pd.DataFrame):

    plt.boxplot(df)
    plt.title("Boxplot Salários Mensais")
    plt.show()

def confusion_matrix(knn, X_test, y_test):
    plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Greens)
    plt.show()
