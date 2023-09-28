
import numpy as np
import pandas as pd
from config import *
from sklearn.preprocessing import MinMaxScaler 

#to check null values
def null_values_percentage(df):
    per_dict={i:np.round(df[i].isnull().mean(),4) for i in df.columns if df[i].isnull().sum()>=1}
    return per_dict

#droping columns that has more than 80% nan values
def drop_column(df,feas):
    df.drop(feas,axis=1,inplace=True)
    return df

#imputing categorical null values
def replace_nan_mode(df):
    cat_nan=[i for i in df.columns if df[i].dtype=='O' and df[i].isnull().sum()>0]
    for fea in cat_nan:
        df[fea].fillna(df[fea].value_counts().index[0],inplace=True)
    return df

#imputing numerical values
def replace_num_nan(df):
    num_nan=[i for i in df.columns if df[i].dtype!='O' and df[i].isnull().sum()>0]
    for fea in num_nan:
        median=df[fea].median()
        df[fea].fillna(median,inplace=True)
    return df

#categorical encoding
def cat_ordinal(df):
    cat_fea=[i for i in df.columns if df[i].dtype=='O']
    for fea in cat_fea:
        rank={k:x for x,k in enumerate(df[fea].value_counts(ascending=True).index,0)}
        df[fea]=df[fea].map(rank)
    return df

#Log transformation
def log_trans(df):
    li=[fea for fea in df.columns if fea in ['LotFrontage','LotArea','1stFlrSF','GrLivArea']]
    for fea in li:
        df[fea]=np.log(df[fea])
    return df

#feature scaling - normalization
def scaling_norm(df):
    feature_scale=[fea for fea in df.columns if fea not in ['Id','SalePrice']]
    mm_scaler=MinMaxScaler()
    mm_scaler.fit(df[feature_scale])
    df=pd.concat([df[['SalePrice']].reset_index(drop=True),pd.DataFrame(mm_scaler.transform(df[feature_scale]),columns=feature_scale)],axis=1)
    return df