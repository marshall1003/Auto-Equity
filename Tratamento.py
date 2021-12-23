import pandas as pd
from sklearn import preprocessing
def enumerar(array):
    le = preprocessing.LabelEncoder()
    le.fit(array)
    return le.transform(array)

def enumerar_dataframe(df_origem: pd.DataFrame):
    
    df = df_origem.copy()
    for i in df.columns:
        df[i].replace(df[i].unique(), enumerar(df[i].unique()), inplace=True)
    return df

def de_para_numeros(array):
    le = preprocessing.LabelEncoder()
    le.fit(array)
    output = {}
    values = le.transform(array)
    keys = le.inverse_transform(le.transform(array))
    for cont in range(len(keys)):
        output[keys[cont]] = values[cont]
    
    return output
