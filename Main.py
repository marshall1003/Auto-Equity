import pandas as pd
import Constants
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import Tratamento
import Graphics
from sklearn.linear_model import LinearRegression

def main ():
    with open("DataBase.csv", "r") as dbfile:
        df = pd.read_csv(dbfile)

    #remoção de dados que poderiam causar ruidos na analise (nulos, colunas desnecessárias, etc.)
        
    df.dropna(subset=[Constants.OPERATION_STATUS, Constants.STATE, Constants.MONTHLY_INCOME], inplace=True)
    df.drop(df.loc[df[Constants.OPERATION_STATUS] == Constants.ONGOING].index, inplace=True)
    df[Constants.CPF_RESTRICTION].replace([True, False], ["True","False"], inplace=True)
    df[Constants.CPF_RESTRICTION].fillna(Constants.NOT_DEFINED, inplace=True)
    Tratamento.idade_df(df)
    print(df["age"])
    Tratamento.adjust_money_to_float(df, Constants.LOAN_AMOUNT)
    Tratamento.adjust_money_to_float(df, Constants.AUTO_VALUE)
    
    #DataFrame convertido inteiramente para valores numéricos
    df_tratado = Tratamento.enumerar_dataframe(df[[Constants.CPF_RESTRICTION,Constants.STATE,Constants.MONTHLY_INCOME, 
    Constants.AUTO_VALUE_FAIXA_VALOR,Constants.LOAN_AMOUNT_VALUE, Constants.OPERATION_STATUS]])
    analise_correl(Constants.CPF_RESTRICTION, df_tratado, df[[Constants.CPF_RESTRICTION, Constants.OPERATION_STATUS]])
    print("Analise estados")
    analise_correl(Constants.STATE, df_tratado, df[[Constants.STATE, Constants.OPERATION_STATUS]])
    
    #Analise de estado pós ajuste por regiao
    Tratamento.estados_por_regiao(df)
    df_tratado = Tratamento.enumerar_dataframe(df[[Constants.CPF_RESTRICTION,Constants.STATE,Constants.MONTHLY_INCOME,Constants.IDADE, 
    Constants.AUTO_VALUE_FAIXA_VALOR,Constants.LOAN_AMOUNT_VALUE, Constants.OPERATION_STATUS]])
    print("Analise regiões")
    analise_correl(Constants.STATE, df_tratado, df[[Constants.STATE, Constants.OPERATION_STATUS]])
    
    #Analise de boxplot para determinar quais faixas arbitrarias iremos dividir os salários.
    Tratamento.adjust_money_to_float(df, Constants.MONTHLY_INCOME)
    df.dropna(subset=[Constants.MONTHLY_INCOME], inplace=True)
    analise_correl(Constants.MONTHLY_INCOME, df_tratado, df[[Constants.MONTHLY_INCOME, Constants.OPERATION_STATUS]])

    print(df_tratado)
    #Boxplot de renda
    message = "quer ver o boxplot?\n1- Sim\n2- Não\n"
    response = input(message)
    cond = True
    while(cond):
        if(response == "1"):
            Graphics.boxplot(df[Constants.MONTHLY_INCOME])
            cond = False
        elif (response == "2"):
            cond = False
        else:
            print("Valor invalido! Digite 1 para Sim e 2 para Não")

    #Predição
    x = df_tratado[[Constants.CPF_RESTRICTION,Constants.STATE,Constants.LOAN_AMOUNT_VALUE, Constants.IDADE, Constants.AUTO_VALUE_FAIXA_VALOR, Constants.MONTHLY_INCOME]]
    y = df_tratado[Constants.OPERATION_STATUS]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=1, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)
    score_predicao = knn.score(X_test, y_test)
    
    #Acuracia
    print("Precisão de",(score_predicao*100).round(2),"%")

    #Plotagem
    Graphics.confusion_matrix(knn, X_test, y_test)

    #Regressão Linear

    regression = LinearRegression()
    regression.fit(X_train, y_train)
    reg_coefs = regression.coef_
    for cont in range(len(reg_coefs)):
        print({x.columns[cont]:reg_coefs[cont]})   
    
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