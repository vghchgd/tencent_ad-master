# coding=utf-8

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.externals import joblib
from scipy import sparse
from sklearn.model_selection import RandomizedSearchCV
import os
from sklearn.metrics import roc_auc_score
import pickle

sample_path = '../data/sample_data/'
raw_path = '../data/raw_data/'
f = open(sample_path + 'train_x_sample.pkl', 'rb')
train_x = pickle.load(f)
f.close()
train_y = pd.read_csv(sample_path + 'train_y_sample.csv')
print (train_x.shape, train_y.shape)

#X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, train_size=0.1, random_state=2018)
X_train = train_x
y_train = train_y
del train_x,train_y


clf = xgb.XGBClassifier(
        boosting_type='gbdt',
        n_estimators=1000,
        learning_rate=0.1, random_state=2020, n_jobs=-1,metric='auc')

# 网格搜索的参数空间
params = {
    'colsample_bytree': [0.7,0.8,0.9],
    'subsample': [0.7,0.8,0.9],
    'min_child_weight' : np.arange(20, 40, 4),
    'reg_alpha' : [0.2, 0.4, 0.8, 1],
    'reg_lambda' : [0.2, 0.4, 0.8, 1],
    'max_depth':[7,8,9],
    'gamma':[0,0.1,0.2,0.3],
    #'bagging_fraction':[0.7,0.8,0.9,1],
    #'bagging_freq':np.arange(5, 10, 1),
}
print('------把train_x_csr分出来百分之10，同样的train_y分出来百分之10作为底下的基础数据------')
# 迭代次数为5次，cv=5进行5折交叉验证，总共运行25轮
rand_search = RandomizedSearchCV(clf, params, cv=5, n_iter=15, random_state=2018, verbose=2)
rand_search.fit(X_train, np.array(y_train).squeeze())
best = rand_search.best_estimator_  # 获取最佳的模型
print(rand_search.best_params_)  # 输出最佳模型的参数
print(rand_search.best_score_)

#res['score'] = best.predict_proba(X_test)[:, 1]
#print('------用初始分出来的测试集看看这个best的auc性能------')
#test_auc = roc_auc_score(res['label'], res['score'].values)
#print("offline auc: ", test_auc)
#res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
#res.to_csv('../datawater/submission4.csv', index=False)
#fea_imp = best.feature_importances_  # 输出最佳模型的特征重要性
#fea_imp = pd.Series(fea_imp)
#fea_imp.to_csv('../datawater/fea_imp.csv', index=False)  # 保存至文件
joblib.dump(best, '../data/model/random_search_best.model')


#{'subsample': 0.9, 'reg_lambda': 0.4, 'reg_alpha': 1, 'min_child_weight': 32, 'max_depth': 9, 'gamma': 0.3, 'colsample_bytree': 0.7}

