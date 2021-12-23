from numpy import string_
from numpy.lib.function_base import gradient
import pandas as pd
import Constants
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import Tratamento
import Graphics
def main ():
    with open("DataBase.csv", "r") as dbfile:
        df = pd.read_csv(dbfile)

    #remoção de dados que poderiam causar ruidos na analise (nulos, colunas desnecessárias, etc.)
        
    df.dropna(subset=[Constants.OPERATION_STATUS, Constants.STATE, Constants.MONTHLY_INCOME], inplace=True)
    df.drop(df.loc[df[Constants.OPERATION_STATUS] == Constants.ONGOING].index, inplace=True)
    df[Constants.CPF_RESTRICTION].replace([True, False], ["True","False"], inplace=True)
    df[Constants.CPF_RESTRICTION].fillna(Constants.NOT_DEFINED, inplace=True)
    Tratamento.idade_df(df)
    
    #DataFrame convertido inteiramente para valores numéricos
    df_tratado = Tratamento.enumerar_dataframe(df[[Constants.CPF_RESTRICTION,Constants.STATE,Constants.MONTHLY_INCOME, 
    Constants.AUTO_VALUE,Constants.LOAN_AMOUNT, Constants.OPERATION_STATUS]])
    analise_correl(Constants.CPF_RESTRICTION, df_tratado, df[[Constants.CPF_RESTRICTION, Constants.OPERATION_STATUS]])
    print("Analise estados")
    analise_correl(Constants.STATE, df_tratado, df[[Constants.STATE, Constants.OPERATION_STATUS]])
    
    #Analise de estado pós ajuste por regiao
    Tratamento.estados_por_regiao(df)
    df_tratado = Tratamento.enumerar_dataframe(df[[Constants.CPF_RESTRICTION,Constants.STATE,Constants.MONTHLY_INCOME, 
    Constants.AUTO_VALUE,Constants.LOAN_AMOUNT, Constants.OPERATION_STATUS]])
    print("Analise regiões")
    analise_correl(Constants.STATE, df_tratado, df[[Constants.STATE, Constants.OPERATION_STATUS]])
    
    #Analise de boxplot para determinar quais faixas arbitrarias iremos dividir os salários.
    df[Constants.MONTHLY_INCOME] = df[Constants.MONTHLY_INCOME].apply(lambda x: float(x.replace(".","").replace(",",".").replace("$","")))
    df.dropna(subset=[Constants.MONTHLY_INCOME], inplace=True)
    Graphics.boxplot(df[Constants.MONTHLY_INCOME])
    analise_correl(Constants.MONTHLY_INCOME, df_tratado, df[[Constants.MONTHLY_INCOME, Constants.OPERATION_STATUS]])

    x = df[[Constants.CPF_RESTRICTION,Constants.STATE, Constants.MONTHLY_INCOME]]
    y = df[Constants.OPERATION_STATUS]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)
    Y_pred_prob = knn.predict_proba(X_test)

    print(Y_pred_prob[5:10])

def analise_correl(parametro, df_tratado: pd.DataFrame, df: pd.DataFrame):
    temp_df = df_tratado[[parametro, Constants.OPERATION_STATUS]]
    if(parametro == Constants.CPF_RESTRICTION):
        temp_df.drop(temp_df.loc[temp_df[parametro] == 1].index, inplace=True)
    depara_operation_status = Tratamento.de_para_numeros(df[Constants.OPERATION_STATUS])
    depara_parametro = Tratamento.de_para_numeros(df[parametro])
    print("número de entradas: " + str(temp_df.shape[0]))
    print(depara_operation_status)
    print(depara_parametro)
    print (temp_df.corr().round(2))
    

    
if __name__ ==  "__main__":
    main()