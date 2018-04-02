# -*- coding: UTF-8 -*-
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import skew


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5


RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


def create_submission(prediction, score):
    now = datetime.datetime.now()
    print type(prediction)
    sub_file = 'submission_baobao_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)


# train need to be test when do test prediction
def read_data():
    train = pd.read_csv('../../input/train.csv')
    test = pd.read_csv('../../input/test.csv')
    list_te_user = []
    for i in test.icol(0):
        list_te_user.append(i)
    #获得特征
    X=train.drop(['SalePrice','Id'],axis=1)
    X_te = test.drop(['Id'],axis=1)
    #标签,进行log
    y=np.log1p(train.SalePrice)

    X_train,y_train = X,y

    X_train_data = X_train.select_dtypes(include=[np.number]).interpolate().dropna()

    X_te_data = X_te.select_dtypes(include=[np.number]).interpolate().dropna()
    return X_train_data,X_te_data,y_train



def model_gradient_boosting_tree(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain
    gbr = GradientBoostingRegressor(random_state=0)
    param_grid = {
        'n_estimators': [800, 1500],
        'max_features': [20, 15],
        'max_depth': [8, 10],
        'learning_rate': [0.1],
        'subsample': [1]
    }
    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_


# read data, build model and do prediction
train = pd.read_csv("../../input/train.csv")  # read train data
test = pd.read_csv("../../input/test.csv")  # read test data
Xtrain, Xtest, ytrain = read_data()


test_predict, score = model_gradient_boosting_tree(Xtrain, Xtest, ytrain)

print test_predict
create_submission(np.exp(test_predict),score)



