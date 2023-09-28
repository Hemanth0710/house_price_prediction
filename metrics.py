from sklearn.metrics import r2_score,mean_absolute_error
from model import *

def predict(X_test,model):
    return model.predict(X_test)

#r2_score
def r2(y_pred,y_test):
    r2=r2_score(y_pred,y_test)
    return r2

#r2_adjusted
def r2_adj(r2):
    return 1-((1-r2)*(30-1)/(30-1-1))

#MAE 
def mae(y_pred,y_test):
    m=mean_absolute_error(y_pred,y_test)
    return m
