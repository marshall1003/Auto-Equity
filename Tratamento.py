import pandas as pd
from sklearn import preprocessing
import Constants
import datetime

def adjust_money_to_float(df:pd.DataFrame, parameter):
    df[parameter] = df[parameter].apply(lambda x: float(x.replace(".","").replace(",",".").replace("$","")))

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
        if(i  == Constants.IDADE or i  == Constants.LOAN_AMOUNT_VALUE or i  == Constants.AUTO_VALUE_FAIXA_VALOR):
            pass
        else:
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

def estados_por_regiao(df: pd.DataFrame):
    df.replace(["RS", "SC", "PR"], ["SUL","SUL", "SUL"], inplace=True)
    df.replace(["SP", "RJ", "MG", "ES"], ["SUDESTE", "SUDESTE", "SUDESTE", "SUDESTE"], inplace=True)
    df.replace(["MT", "MS", "GO", "DF"], ["CO","CO", "CO", "CO"], inplace=True)
    df.replace(["AC", "AM", "RR", "RO", "PA", "AP", "TO"], ["NORTE","NORTE","NORTE","NORTE", "NORTE","NORTE", "NORTE"], inplace=True)
    df.replace(["MA", "PI", "BA", "CE", "PE", "AL", "SE", "PB", "RN"], ["NORDESTE","NORDESTE","NORDESTE","NORDESTE", "NORDESTE","NORDESTE", "NORDESTE", "NORDESTE", "NORDESTE"], inplace=True)
    