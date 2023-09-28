from eda import *
from model import *
from metrics import *
from config import *
from sklearn.model_selection import train_test_split
import pandas as pd

def process(df):
    #checking null values
    per_dict=null_values_percentage(df)
    print(per_dict)

    #droping columns
    li_nan=[fea for fea in per_dict.keys() if per_dict[fea]>0.8]
    df=drop_column(df,li_nan)
    df=drop_column(df,[fea for fea in df.columns if 'Id' in fea])

    #imputing numerical fea
    df=replace_num_nan(df)

    #imputing categorical fea
    df=replace_nan_mode(df)

    #categorical Encoding
    df=cat_ordinal(df)
    
    #Transforming the some numerical features
    df=log_trans(df)

    #Scaling the features
    df=scaling_norm(df)

    X=df.drop(['SalePrice'],axis=1)
    y=df.SalePrice

    #splitting the data
    X_train,X_test,y_train,y_test = train_test_split(X,y)

    #model training
    lin_reg=linear_regressor(X_train,y_train)

    #predict
    lin_pred=predict(X_test,lin_reg)

    #metric score
    r2_lin=r2(lin_pred,y_test)
    ma=mae(lin_pred,y_test)

    print("r2 : ",r2_lin)
    print("MAE : ",ma)

def main():
    df1=pd.read_csv(hsp_path)
    process(df1)
    
if __name__ == "__main__":
    main()

    
