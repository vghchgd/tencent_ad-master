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
import pickle

sample_feature_data_path = '../data/count_feat/'
raw_data_path = '../data/raw_data/'

def gen_data():

    print ('-------------------------read data------------------------')
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


    train=pd.read_csv('../data/train.csv')
    predict=pd.read_csv('../data/test2.csv')
    train.loc[train['label']==-1,'label']=0
    predict['label']=-1
    data=pd.concat([train,predict])
    data=pd.merge(data,ad_feature,on='aid',how='left')
    data=pd.merge(data,user_feature,on='uid',how='left')
    data=data.fillna('-1')
    train = data[data['label']==-1]

    drop_feature =['appIdAction','appIdInstall','interest3','interest4','kw3','topic3']
    data.drop(drop_feature, axis=1,inplace=True)
    return data


def gen_ct(data):
    print ('-------------------------gen ct data------------------------')
    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    ct_train = data['ct'].values
    ct_train = [m.split(' ') for m in ct_train]
    ct_trains = []
    for i in ct_train:
        index = [0, 0, 0, 0, 0]
        for j in i:
            index[int(j)] = 1
        ct_trains.append(index)

    ct_feat = pd.DataFrame(ct_trains, columns=['ct_{}'.format(i) for i in range(5)])
    data = pd.concat([data, ct_feat], axis=1)
    return data	

def ctr_imp_read():
    ctr_imp = []
    with open('../data/imp_ctr_tang.txt', 'r') as f:
        for i in range(30):
            line = f.readline()
            ctr_imp.append(line.strip().split(' ')[1])
    return ctr_imp


def ctr_feat(data):
    singlefea=[ ##下面是test_auc和test_acu1共有的
            'education','creativeId','campaignId','adCategoryId',
            'carrier','productId','house',
            'age','creativeSize','marriageStatus',
           ##下面是test_auc前30名独有的
            'gender',
            ##下面是test_auc1前30名独有的
            'productType', 'advertiserId']

    pairfea=[##下面是test_auc和test_acu1共有的
        ('aid', 'age'),('creativeId', 'age'),
        ('campaignId', 'age'),('campaignId', 'gender'),
        ('advertiserId', 'age'),
        ('adCategoryId', 'age'),
          ('productId', 'age'),
         ('adCategoryId', 'gender'),
        ('advertiserId', 'gender'),
       ('aid','gender'),
        ('creativeId', 'gender'),
        ('productType', 'house'),
       ('productType', 'age'),
         ('productType', 'marriageStatus'),
        ('productType', 'education'),
##下面是test_auc1前30名独有的
        ('productId', 'gender'),
        ('productType','gender'),
        ('adCategoryId','education')]

    def file_name(file_dir):
        L = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                L.append(file)
        return L

    sing_ctr_path = '../data/single_ctr/'
    cross_ctr_path = '../data/cross_ctr/'
    # ctr_path = 'C:/Users/opt_12/PycharmProjects/data/ctr_top30/'
    #sing_ctr_imp = file_name(sing_ctr_path)
    #cross_ctr_imp = file_name(cross_ctr_path)
    for fea1 in singlefea:
        fea1ctr=pd.read_csv('%s.csv' % (sing_ctr_path + fea1))   #singlefea_ctr的路径
        data = pd.merge(data, fea1ctr, how='left', on=fea1ctr.columns[0])
        data.fillna(value=fea1ctr[fea1ctr.columns[-1]].mean(), inplace=True)
        print('------{} merge Down------'.format(fea1))
    for fea1,fea2 in pairfea:
        fea1ctr=pd.read_csv('%s.csv' % (cross_ctr_path + fea1 +'_'+fea2+ 'ctr'))   #pairfea_ctr的路径
        data = pd.merge(data, fea1ctr, how='left', on=[fea1,fea2])
        data.fillna(value=fea1ctr[fea1ctr.columns[-1]].mean(), inplace=True)
        print('------{}_{}merge Down------'.format(fea1, fea2))

    return data


def func(x1, x2):
    return str(x1)+'_'+str(x2)

def func2(x1, x2, x3):
    return str(x1)+'_'+str(x2)+"_"+str(x3)


# In[14]:


def cross_feat(train, fea1, fea2):
    value = list(map(func, train[fea1], train[fea2]))
    feat = pd.Series(value, name='{}_{}'.format(fea1, fea2))
    return feat

def cross_3feat(train, fea1, fea2, fea3):
    value = list(map(func2, train[fea1], train[fea2], train[fea3]))
    feat = pd.Series(value, name='{}_{}_{}'.format(fea1, fea2, fea3))
    return feat


def add_cross(data):
    
    print ('-------------------------gen cross data------------------------')
    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    cross_2fea = [('productType', 'creativeSize'),
     ('productType', 'marriageStatus'),
     ('productType', 'os'),
     ('age', 'education'),
     ('age', 'carrier'),
     ('education', 'carrier'),
     ('creativeSize', 'carrier'),
     ('consumptionAbility', 'carrier'),
     ('marriageStatus', 'os')
     ]
    cross_3fea = [('education', 'marriageStatus', 'productType'),('education', 'marriageStatus', 'carrier'),  ('education', 'age', 'carrier')]

    vector_feature= ['interest1','interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']

    one_hot_feature=['age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
           'adCategoryId', 'productId', 'productType']

    for fea1, fea2, fea3 in cross_3fea:
         fea1_fea2_fea3 = cross_3feat(data, fea1, fea2, fea3)
         data[fea1_fea2_fea3.name] = fea1_fea2_fea3
         vector_feature.append((fea1_fea2_fea3.name))
    for fea1, fea2 in cross_2fea:
         fea1_fea2 = cross_feat(data, fea1, fea2)
         data[fea1_fea2.name] = fea1_fea2
         one_hot_feature.append(fea1_fea2.name)

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('-----------------------LabelEncoder feature data Down-----------')

    output = open(raw_data_path+ "LabelEncode_data_519.pkl", 'wb')
    pickle.dump(data, output, protocol=4)
    output.close()
    print('-----------------------LabelEncoder feature data Saved-----------')    


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)



# In[21]:
def target_enc(train, test, train_y):
    gc.collect()
    train_encoded, test_encoded = target_encode(trn_series=train["LBS"], tst_series=test["LBS"],
                                    target=train_y,
                                    min_samples_leaf=100,
                                    smoothing=10,
                                    noise_level=0.01)
    train['LBS_enc'] = train_encoded
    # train.drop('LBS', axis=1, inplace=True)
    test['LBS_enc'] = test_encoded
    # test.drop('LBS', axis=1, inplace=True)
  
    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('-----------------------target_encode feature data Down-----------')
    return train, test


def xgb_leaf(train, test):
    # train_xgb = pd.read_csv('../data/train_topN_and_xgb_feature.csv')
    test_xgb = pd.read_csv('../data/test_topN_and_xgb_feature_519.csv')
    # train = pd.concat([train, train_xgb], axis=1)
    test = pd.concat([test, test_xgb], axis=1)
    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('-----------------------xgb_leaf data concat Down-----------')
    # output = open(raw_data_path+ "xgb_leaf_train_519.pkl", 'wb')
    # pickle.dump(train, output, protocol=4)
    # output.close()
    output = open(raw_data_path+ "xgb_leaf_test_519.pkl", 'wb')
    pickle.dump(test, output, protocol=4)
    output.close()
    print('-----------------------xgb_leaf feature data Saved-----------')    
    return train, test

def one_hot_gen(train, test):
    
    

    vector_feature= ['interest1','interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2','education_marriageStatus_productType','education_marriageStatus_carrier', 'education_age_carrier']

    one_hot_feature=['productType_creativeSize',
     'productType_marriageStatus',
     'productType_os',
     'age_education','age_carrier',
     'education_carrier','creativeSize_carrier','consumptionAbility_carrier',
     'marriageStatus_os']
    #for i in range(30):
    #    one_hot_feature.append(str(i))
    fea_list = []
    for i in train.columns:
        if i not in one_hot_feature and i not in vector_feature and i not in ['aid', 'uid']: 
            fea_list.append(i) 
    print(fea_list)


    print('-----------------------start One-hot encode------------------')
    enc = OneHotEncoder()
    train['LBS'] = train['LBS'].astype(int)
    test['LBS'] = test['LBS'].astype(int)
    data = pd.concat([train, test])
    train_x=train[fea_list].values
    test_x=test[fea_list].values

    for feature in one_hot_feature:
        print(feature)
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a=enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x= sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('one-hot prepared !')
    sparse.save_npz('../data/train_x_oh_22_ctrtop20.npz', train_x)
    sparse.save_npz('../data/test_x_oh_22_ctrtop20.npz', test_x)
    del train_x, test_x
    print('-----------------------one hot feature data Saved-----------')


def vector_gen(data, train, test, train_x, test_x):
    vector_feature= ['interest1','interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2','education_marriageStatus_productType','education_marriageStatus_carrier', 'education_age_carrier']

    cv=CountVectorizer(min_df=0.0012)
    for feature in vector_feature:
        print(feature)
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('cv prepared !')
    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

  
    sparse.save_npz('../data/train_x_20_ctrtop38.npz', train_x)
    sparse.save_npz('../data/test_x_20_ctrtop38.npz', test_x)

    print('data saved !')



def XGB_predict(train_x,train_y,test_x,res):
    print("XGB test")
    clf = XGBClassifier(    
                        n_estimators=1000,
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
                        seed=2020
                     )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=50)
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('../data/submission_xgb_add_feat.csv', index=False)
    os.system('zip ../data/baseline_xgb_add_feat.zip ../data/submission_xgb_add_feat.csv')
    return clf






if __name__=='__main__':

    # data = gen_data()
    # data = gen_ct(data)
    # add_cross(data)

    #f = open(raw_data_path + 'LabelEncode_data_519.pkl', 'rb')
    #data = pickle.load(f)
    #f.close()
    #train=data[data.label!=-1]
    #train.pop('label')
    #train_y.to_csv('../data/train_y_519.csv', index=False)
    #test=data[data.label==-1]
    # train, test = target_enc(train, test, train_y)
    # print(test.shape)
    # train, test = xgb_leaf(train, test)
    # print(test.shape)






    # f = open(raw_data_path + 'xgb_leaf_train_519.pkl', 'rb')
    # train = pickle.load(f)
    # f.close()
    #
    # f = open(raw_data_path + 'xgb_leaf_test_519.pkl', 'rb')
    # test = pickle.load(f)
    # f.close()
    #

    #train = ctr_feat(train)
    
    #output = open(raw_data_path + "ctr_train_top20.pkl", 'wb')
    #pickle.dump(train, output, protocol=4)
    #output.close()
    
    
    #test = ctr_feat(test)
    #print(test.shape)
    #output = open(raw_data_path + "ctr_test_top20.pkl", 'wb')
    #pickle.dump(test, output, protocol=4)
    #output.close()

    #one_hot_gen(train, test)
    # f = open(raw_data_path + 'ctr_train_520.pkl', 'rb')
    # train = pickle.load(f)
    # f.close()
    #
    # f = open(raw_data_path + 'ctr_test_520.pkl', 'rb')
    # test = pickle.load(f)
    # f.close()
    #
    # one_hot_gen(train, test)

    


    #

    #f = open(raw_data_path + 'ctr_train_top38.pkl', 'rb')
    #train = pickle.load(f)
    #f.close()
    #train.drop('label', axis=1, inplace=True)
    #f = open(raw_data_path + 'ctr_test_top38.pkl', 'rb')
    #test = pickle.load(f)
    #f.close()
    #test.drop('label', axis=1, inplace=True)
    #one_hot_gen(train, test)

    train_x = sparse.load_npz('../data/train_x_oh_22_ctrtop38.npz')
    test_x = sparse.load_npz('../data/test_x_oh_22_ctrtop38.npz')
    
    raw_train = sparse.load_npz('../data/model_fea_add3_train.npz')
    raw_test = sparse.load_npz('../data/model_fea_add3_test.npz')
    print(train_x.shape, test_x.shape, raw_train.shape, raw_test.shape)
    train_x = sparse.hstack((train_x, raw_train))
    test_x = sparse.hstack((test_x, raw_test))

    sparse.save_npz('../data/train_x_22_ctrtop20.npz', train_x)
    sparse.save_npz('../data/test_x_22_ctrtop20.npz', test_x)
    
    #vector_feature= ['interest1','interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2','education_marriageStatus_productType','education_marriageStatus_carrier', 'education_age_carrier']
    #train = train[vector_feature]
    #test = test[vector_feature]
    #data = pd.concat([train, test])
    #vector_gen(data, train, test, train_x, test_x)
    #del train,test
    #del data

    #train_x = sparse.load_npz(raw_path + 'train_x_20_ctrtop10.npz')
    #test_x =  sparse.load_npz(raw_path + 'test_x_20_ctrtop10.npz')
    #train_y = pd.read_csv('../data/train_y_518.csv', header=None)
    #test = pd.read_csv('../data/test2.csv')
    #res=test[['aid','uid']]
    #model=XGB_predict(train_x,train_y,test_x,res)
    #fea_imp = pd.Series(model.feature_importances_)
    #fea_imp.to_csv('../data/fea_imp_003.csv', index=False)
    #joblib.dump(model, '../data/model/xgb_003.model')
    #print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))




    





