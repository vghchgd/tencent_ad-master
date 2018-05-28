import pandas as pd
import numpy as np
#import lightgbm as lgb
#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import datetime

print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
ad_feature=pd.read_csv('../data/adFeature.csv')
if os.path.exists('../data/userFeature.csv'):
    user_feature=pd.read_csv('../data/userFeature.csv')
else:
    userFeature_data = []
    with open('../data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('../data/userFeature.csv', index=False)
        del userFeature_data
print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
train=pd.read_csv('../data/train.csv')
predict=pd.read_csv('../data/test1.csv')
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
#data=data.fillna('-1')

data.sample(frac = 0.01).to_csv('../data/data_sample.csv')

