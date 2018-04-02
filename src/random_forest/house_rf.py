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
    sub_file = 'submission_' + str(score) + '_' + 'baobao_rf'+ '.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)


def data_preprocess(train, test):
    outlier_idx = [4, 11, 13, 20, 46, 66, 70, 167, 178, 185, 199, 224, 261, 309, 313, 318, 349, 412, 423, 440, 454, 477,
                   478, 523, 540, 581, 588, 595, 654, 688, 691, 774, 798, 875, 898, 926, 970, 987, 1027, 1109, 1169,
                   1182, 1239, 1256, 1298, 1324, 1353, 1359, 1405, 1442, 1447]
    train.drop(train.index[outlier_idx], inplace=True)
    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

    to_delete = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    all_data = all_data.drop(to_delete, axis=1)

    train["SalePrice"] = np.log1p(train["SalePrice"])
    # log transform skewed numeric features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    print "data done --------------------------"
    return X_train, X_test, y


def model_random_forecast(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain
    rfr = RandomForestRegressor(n_jobs=1, random_state=0)
    param_grid = {'n_estimators': [100,200,400,500,1000], 'max_features': [10,15,20,25], 'max_depth':[20,20,25,25,]}
    # 'n_estimators': [1000], 'max_features': [10,15,20,25], 'max_depth':[20,20,25,25,]}
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Random forecast regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_


# read data, build model and do prediction
# read train data
train = pd.read_csv("../../input/train.csv")
# read test data
test = pd.read_csv("../../input/test.csv")
Xtrain, Xtest, ytrain = data_preprocess(train, test)

test_predict, score = model_random_forecast(Xtrain, Xtest, ytrain)

create_submission(np.exp(test_predict), score)



