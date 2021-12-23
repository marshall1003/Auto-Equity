import pandas as pd
from sklearn import preprocessing
import Constants
import datetime

def idade_df (df: pd.DataFrame):
    temp = pd.DataFrame()
    temp["dob"] = pd.to_datetime(df["birth_date"])
    now = pd.datetime.now()
    df["age"] = temp["dob"].apply(lambda x : (now.year - x.year))
    return df

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

def faixas_salariais(df: pd.DataFrame):
    temp_df = df.copy()
    #temp_df.replace(temp_df.loc[temp_df[Constants.MONTHLY_INCOME] >= 800  ], 0, inplace=True)
    #temp_df.replace(temp_df.loc[temp_df[Constants.MONTHLY_INCOME] >= 1600 ], 1, inplace=True)
    #temp_df.replace(temp_df.loc[temp_df[Constants.MONTHLY_INCOME] >= 2400 ], 2, inplace=True)
    #temp_df.replace(temp_df.loc[temp_df[Constants.MONTHLY_INCOME] >= 3200 ], 3, inplace=True)
    #temp_df.replace(temp_df.loc[temp_df[Constants.MONTHLY_INCOME] >= 4000 ], 4, inplace=True)
    #temp_df.replace(temp_df.loc[temp_df[Constants.MONTHLY_INCOME] >= 5500 ], 5, inplace=True)
    #temp_df.replace(temp_df.loc[temp_df[Constants.MONTHLY_INCOME] >= 7000 ], 6, inplace=True)
    #temp_df.replace(temp_df.loc[temp_df[Constants.MONTHLY_INCOME] >= 10000], 7, inplace=True)

    return temp_df
    
def estados_por_regiao(df: pd.DataFrame):
    df.replace(["RS", "SC", "PR"], ["SUL","SUL", "SUL"], inplace=True)
    df.replace(["SP", "RJ", "MG", "ES"], ["SUDESTE", "SUDESTE", "SUDESTE", "SUDESTE"], inplace=True)
    df.replace(["MT", "MS", "GO", "DF"], ["CO","CO", "CO", "CO"], inplace=True)
    df.replace(["AC", "AM", "RR", "RO", "PA", "AP", "TO"], ["NORTE","NORTE","NORTE","NORTE", "NORTE","NORTE", "NORTE"], inplace=True)
    df.replace(["MA", "PI", "BA", "CE", "PE", "AL", "SE", "PB", "RN"], ["NORDESTE","NORDESTE","NORDESTE","NORDESTE", "NORDESTE","NORDESTE", "NORDESTE", "NORDESTE", "NORDESTE"], inplace=True)
    