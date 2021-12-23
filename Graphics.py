import pandas as pd
import matplotlib.pyplot as plt

def boxplot(df:pd.DataFrame):

    plt.boxplot(df)
    plt.title("Boxplot Salários Mensais")
    plt.show()

