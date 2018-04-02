# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:27:25 2017

@author: H.P. Asela
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from keras.models import Sequential
from keras.layers import Dense

def create_submission(prediction):
    # now = datetime.datetime.now()
    sub_file = 'submission_' + 'nn' + '.csv'
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

X_train,X_test,y = read_data()

train = pd.read_csv("../../input/train.csv")  # read train data
test = pd.read_csv("../../input/test.csv")  # read test data

model = Sequential()
model.add(Dense(100, input_dim=36, kernel_initializer='normal', activation='relu'))
model.add(Dense(40, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(np.array(X_train), np.array(y), nb_epoch=20, batch_size=50)
predictions = model.predict(np.array(X_test))

list_pre = []
for i in np.expm1(predictions):
    print float(i)
    list_pre.append(float(i))
create_submission(list_pre)

# np.savetxt('predict_baobao.csv', predictions, delimiter="\t")