
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn import cross_validation, metrics
import scipy.special as special
from sklearn.externals import joblib
import gc, os
import datetime
import pickle


# In[23]:


test = pd.read_csv('../data/test1.csv')


# In[24]:


res=test[['aid','uid']]


# In[21]:


train_x = sparse.load_npz('../data/train_x.npz')
test_x =  sparse.load_npz('../data/test_x.npz')
train_y = pd.read_csv('../data/train_y.csv', )


# In[20]:


def LGB_predict(train_x,train_y,test_x,res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
         boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=5000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)

    feat_imp = clf.feature_importances_
#     print(feat_imp)
    fea_imp = pd.Series(fea_imp)
    fea_imp.to_csv('./datasets/fea_imp.csv', index=False)  
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./datasets/submission_{}.csv'.format(timedate), index=False)
    os.system('zip ./datasets/baseline_{}.zip ./datasets/submission_{}.csv'.format(timedate, timedate))
    return clf

model = LGB_predict(train_x,train_y,test_x,res)
joblib.dump(model, './datasets/lgb_submit.model')


# In[8]:




