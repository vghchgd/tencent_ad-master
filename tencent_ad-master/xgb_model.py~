# coding=utf-8
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import os
import datetime

raw_path = '../data/raw_data/'

train_x = sparse.load_npz(raw_path + 'train_x_20_ctrtop10.npz')
test_x =  sparse.load_npz(raw_path + 'test_x_20_ctrtop10.npz')
train_y = pd.read_csv('../data/train_y_518.csv', header=None)
test = pd.read_csv('../data/test2.csv')
res=test[['aid','uid']]

print( "---------------load data Down------------" )


def XGB_predict(train_x,train_y,test_x,res):
    print("XGB test")
    clf = XGBClassifier(    
                        n_estimators=2000,
                        max_depth=7,
                        objective="binary:logistic",
                        learning_rate=0.1, 
                        subsample=.8,
                        min_child_weight=5,
                        colsample_bytree=.9,
                        #scale_pos_weight=1.6,
                        gamma=0,
                        #reg_alpha=8,
                        #reg_lambda=1.3,
                        n_jobs=-1,
                        scale_pos_weight=1,
                        seed=2018
                     )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=50)
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('../data/submission_xgb_add_feat.csv', index=False)
    os.system('zip ../data/baseline_xgb_add_feat.zip ../data/submission_xgb_add_feat.csv')
    return clf

model=XGB_predict(train_x,train_y,test_x,res)
fea_imp = pd.Series(model.feature_importances_)
fea_imp.to_csv('../data/fea_imp_003.csv', index=False)
joblib.dump(model, '../data/model/xgb_003.model')
print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
