# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import time
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, LassoCV, LassoLarsCV, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import linear_model,svm
from sklearn.ensemble import BaggingRegressor
from scipy.stats import skew


def create_submission(prediction, score):
    now = datetime.datetime.now()
    print score
    sub_file = 'submission_' + 'baobao3_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    # sub_file = 'prediction_training.csv'
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



def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5


RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


class ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, train, test, ytr):

        X = train.values
        y = ytr.values
        T = test.values
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=0))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))



        for i, reg in enumerate(base_models):
            print ("Fitting the base model...")
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                reg.fit(X_train, y_train)
                #pred list
                y_pred = reg.predict(X_holdout)

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = reg.predict(T)

            print S_train
            '''
            第i个回归器的预测结果（1460,10),这里的１０是交叉验证
            # print S_test_i
            然后取平均，得到结果(1460,1)
            一共有１６个分类器对应回１６个列的位置
            '''

            S_test[:, i] = S_test_i.mean(1)

        print ("Stacking base models...")
        # tuning the stacker
        param_grid = {
            # 'n_estimators': [10 , 50 , 90 , 100 , 110 , 120 , 150 ],
        }
        grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, scoring=RMSE)
        grid.fit(S_train, y)
        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)

        except:
            pass

        y_pred = grid.predict(S_test)


        return y_pred, -grid.best_score_


train = pd.read_csv("../../input/train.csv")  # read train data
test = pd.read_csv("../../input/test.csv")  # read test data

# build a model library (can be improved)
base_models = [
    RandomForestRegressor(
        n_jobs=1, random_state=0,
        n_estimators=160, max_depth=11
    ),
    RandomForestRegressor(
        n_jobs=1, random_state=0,
        n_estimators=150, max_features=12,
        max_depth=11
    ),
    ExtraTreesRegressor(
        n_jobs=1, random_state=0,
        n_estimators=800, max_features=15
    ),
    ExtraTreesRegressor(
        n_jobs=1, random_state=0,
        n_estimators=800, max_features=15
    ),
    GradientBoostingRegressor(
        random_state=0,
        n_estimators=800, max_features=10, max_depth=15,
        learning_rate=0.01, subsample=1
    ),
    GradientBoostingRegressor(
        random_state=0,
        n_estimators=800, max_features=15, max_depth=15,
        learning_rate=0.01, subsample=1
    ),
    XGBRegressor(
        seed=0,
        n_estimators=800, max_depth=15,
        learning_rate=0.01, subsample=1, colsample_bytree=0.75
    ),

    XGBRegressor(
        seed=0,
        n_estimators=800, max_depth=12,
        learning_rate=0.01, subsample=0.8, colsample_bytree=0.75
    ),
    LassoCV(alphas=[1, 0.1, 0.001, 0.0005, 0.0002 ,0.0001, 0.00005]),
    KNeighborsRegressor(n_neighbors=5),
    KNeighborsRegressor(n_neighbors=10),
    KNeighborsRegressor(n_neighbors=15),
    KNeighborsRegressor(n_neighbors=25),
    LassoLarsCV(),
    ElasticNet(),
    SVR()
]

seed = 7
lr = linear_model.LinearRegression()
model = BaggingRegressor(base_estimator=lr, random_state=seed)

ensem = ensemble(
    n_folds=10,
    stacker=model,
    base_models=base_models
)

X_train, X_test, y_train = read_data()
y_pred, score = ensem.fit_predict(X_train, X_test, y_train)
create_submission(np.expm1(y_pred), score)

