import pandas as pd
import Constants
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import Tratamento

def main ():
    with open("DataBase.csv", "r") as dbfile:
        df = pd.read_csv(dbfile)

    #remoção de dados que poderiam causar ruidos na analise (nulos, colunas desnecessárias, etc.)
        
    df.dropna(subset=[Constants.OPERATION_STATUS, Constants.STATE, Constants.MONTHLY_INCOME], inplace=True)
    df.drop(df.loc[df[Constants.OPERATION_STATUS] == Constants.ONGOING].index, inplace=True)
    #df.replace([Constants.CLOSED, Constants.NO_CLOSED], [1,0], inplace=True)
    df[Constants.CPF_RESTRICTION].replace([True, False], ["True","False"], inplace=True)
    df = df[[Constants.CPF_RESTRICTION,Constants.STATE,Constants.MONTHLY_INCOME, 
    Constants.AUTO_VALUE,Constants.LOAN_AMOUNT, Constants.OPERATION_STATUS]]
    df[Constants.CPF_RESTRICTION].fillna(Constants.NOT_DEFINED, inplace=True)
    
    #DataFrame convertido inteiramente para valores numéricos
    df_tratado = Tratamento.enumerar_dataframe(df)
    
    temp_df = df_tratado[[Constants.CPF_RESTRICTION, Constants.OPERATION_STATUS]]
    temp_df.drop(temp_df.loc[temp_df[Constants.CPF_RESTRICTION] == 1].index, inplace=True)
    depara_operation_status = Tratamento.de_para_numeros(df[Constants.OPERATION_STATUS])
    depara_CPF_restriction = Tratamento.de_para_numeros(df[Constants.CPF_RESTRICTION])
    print(depara_operation_status)
    print(depara_CPF_restriction)
    print (temp_df.corr().round(2))
    

    x = df[[Constants.CPF_RESTRICTION,Constants.STATE, Constants.MONTHLY_INCOME]]
    y = df[Constants.OPERATION_STATUS]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)
    Y_pred_prob = knn.predict_proba(X_test)

    print(Y_pred_prob[5:10])



    
if __name__ ==  "__main__":
    main()