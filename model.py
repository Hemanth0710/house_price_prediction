from sklearn.linear_model import LinearRegression  
import warnings
warnings.filterwarnings("ignore")

#multilpleLinear Regression
def linear_regressor(X_train,y_train):
    lr=LinearRegression()
    return lr.fit(X_train,y_train)
