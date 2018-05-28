
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.model_selection import train_test_split
import scipy.special as special
from sklearn.externals import joblib
import gc, os
import datetime
import pickle


# In[ ]:
global data
global train

sample_feature_data_path = '../data/count_feat/'
raw_data_path = '../data/raw_data/'

def gen_data():
    global data
    global train	
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
    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


    # In[2]:


    

    # In[3]:


    train=pd.read_csv('../data/train.csv')
    predict=pd.read_csv('../data/test1.csv')
    train.loc[train['label']==-1,'label']=0
    predict['label']=-1
    data=pd.concat([train,predict])
    data=pd.merge(data,ad_feature,on='aid',how='left')
    data=pd.merge(data,user_feature,on='uid',how='left')
    data=data.fillna('-1')
    train = data[data['label']==-1]

    # In[4]:


    drop_feature = ['appIdAction','appIdInstall','interest3','interest4','kw3','topic3']
    data.drop(drop_feature, axis=1,inplace=True)


# In[7]:


def gen_features():
    global data
    global train	
    # test = load_pickle(raw_data_path + 'sample_test.pkl')
    print("uid_adCount")
    temp = train.groupby('uid')['aid'].count().reset_index()
    temp.columns=['uid','uid_adCount']
    temp.to_csv(sample_feature_data_path+'uid_adCount.csv',index=False)

    for feat_1 in ['creativeId', 'aid', 'advertiserId','campaignId','adCategoryId',
                   'education', 'LBS', 'consumptionAbility','marriageStatus','age','house']:
        gc.collect()
        temp = train[[feat_1, 'label']]
        count = temp.groupby([feat_1]).apply(lambda x: x['label'].count()).reset_index(
            name=feat_1 + '_all')
        count1 = temp.groupby([feat_1]).apply(lambda x: x['label'].sum()).reset_index(
            name=feat_1 + '_1')
        count[feat_1 + '_1'] = count1[feat_1 + '_1']
        count.fillna(value=-1, inplace=True)

        count.to_csv(sample_feature_data_path + '%s.csv' % (feat_1), index=False)
        print(feat_1, ' over')


    for feat_1,feat_2 in [('creativeId','LBS'),('creativeId','age'),('creativeId','gender'),
                          ('creativeId','education'),('creativeId','marriageStatus'),
                          ('creativeId','house'),('creativeId','consumptionAbility'),#('creativeId','uid'),
                          ('aid', 'LBS'), ('aid', 'age'), ('aid', 'gender'),
                          ('aid', 'education'), ('aid', 'marriageStatus'),
                          ('aid', 'house'),('aid','consumptionAbility'),#('aid','uid'),
                          ('advertiserId', 'LBS'), ('advertiserId', 'age'), ('advertiserId', 'gender'),
                          ('advertiserId', 'education'), ('advertiserId', 'marriageStatus'),
                          ('advertiserId', 'house'),('advertiserId','consumptionAbility'),#('advertiserId', 'uid'),
                          ('campaignId', 'LBS'), ('campaignId', 'age'), ('campaignId', 'gender'),
                          ('campaignId', 'education'), ('campaignId', 'marriageStatus'),
                          ('campaignId', 'house'),('campaignId','consumptionAbility'),#('campaignId', 'uid'),
                          ('adCategoryId', 'LBS'), ('adCategoryId', 'age'), ('adCategoryId', 'gender'),
                          ('adCategoryId', 'education'), ('adCategoryId', 'marriageStatus'),
                          ('adCategoryId', 'house'),('adCategoryId','consumptionAbility'),#('adCategoryId', 'uid')
                          ]:
        gc.collect()

        if os.path.exists(sample_feature_data_path + '%s.csv' % (feat_1+'_'+feat_2 )):
            print('found  ' + sample_feature_data_path + '%s.csv' % (feat_1+'_'+feat_2 ) )
        else:
            print('generate ' + sample_feature_data_path + '%s.csv' % (feat_1+'_'+feat_2 ) )

            temp = train[[feat_1, feat_2, 'label']]
            count = temp.groupby([feat_1,feat_2]).apply(lambda x: x['label'].count()).reset_index(
                name=feat_1 + '_' + feat_2 + '_all')
            count1 = temp.groupby([feat_1, feat_2]).apply(lambda x: x['label'].sum()).reset_index(
                name=feat_1 + '_' + feat_2 + '_1')
            count[feat_1 + '_' + feat_2 + '_1'] = count1[feat_1 + '_' + feat_2 + '_1']
            count.fillna(value=0, inplace=True)

            count.to_csv(sample_feature_data_path + '%s.csv' % (feat_1+'_'+feat_2), index=False)
            print(feat_1, feat_2, ' over')


# In[11]:


def add_AllFeatures(data):


    count = pd.read_csv(sample_feature_data_path + 'uid_adCount.csv')
    data = data.merge(count, how='left', on='uid')
    data.fillna(value=0, inplace=True)
    print ('uid over')

    for feat_1 in ['creativeId', 'aid', 'advertiserId', 'campaignId', 'adCategoryId',
                   'education', 'LBS', 'consumptionAbility', 'marriageStatus', 'age', 'house']:
        count = pd.read_csv(sample_feature_data_path + '%s.csv' % (feat_1))
#         bs = BayesianSmoothing(1, 1)
#         bs.update(count[feat_1 + '_all'].values, count[feat_1 + '_1'].values, 1000, 0.001)
#         count[feat_1 + '_ctr'] = (count[feat_1 + '_1'] + bs.alpha) / (
#                 count[feat_1 + '_all'] + bs.alpha + bs.beta)
#         count[feat_1 + '_ctr'] = count[feat_1 + '_ctr'].apply(lambda x: float('%.6f' % x))
        # count.drop([feat_1 + '_1', feat_1 + '_all'], axis=1, inplace=True)
        count[feat_1  + '_ctr'] = 100*(count[feat_1 + '_1'] +0.0001) / (
            count[feat_1  + '_all'] + 0.0001)
        count[feat_1 + '_ctr'] = count[feat_1 + '_ctr'].apply(lambda x: float('%.4f' % x))
        count.drop([feat_1 +  '_1', feat_1 +  '_all'], axis=1, inplace=True)
        data = data.merge(count, how='left', on=feat_1)
#         data.fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)
        data.fillna(value=0, inplace=True)

        print ("concat " + feat_1 + "over")

    for feat_1, feat_2 in [('creativeId', 'LBS'), ('creativeId', 'age'), ('creativeId', 'gender'),
                           ('creativeId', 'education'), ('creativeId', 'marriageStatus'),
                           ('creativeId', 'house'), ('creativeId', 'consumptionAbility'),  # ('creativeId','uid'),
                           ('aid', 'LBS'), ('aid', 'age'), ('aid', 'gender'),
                           ('aid', 'education'), ('aid', 'marriageStatus'),
                           ('aid', 'house'), ('aid', 'consumptionAbility'),  # ('aid','uid'),
                           ('advertiserId', 'LBS'), ('advertiserId', 'age'), ('advertiserId', 'gender'),
                           ('advertiserId', 'education'), ('advertiserId', 'marriageStatus'),
                           ('advertiserId', 'house'), ('advertiserId', 'consumptionAbility'),
                           # ('advertiserId', 'uid'),
                           ('campaignId', 'LBS'), ('campaignId', 'age'), ('campaignId', 'gender'),
                           ('campaignId', 'education'), ('campaignId', 'marriageStatus'),
                           ('campaignId', 'house'), ('campaignId', 'consumptionAbility'),  # ('campaignId', 'uid'),
                           ('adCategoryId', 'LBS'), ('adCategoryId', 'age'), ('adCategoryId', 'gender'),
                           ('adCategoryId', 'education'), ('adCategoryId', 'marriageStatus'),
                           ('adCategoryId', 'house'), ('adCategoryId', 'consumptionAbility'),  # ('adCategoryId', 'uid')
                           ]:
        gc.collect()
        count = pd.read_csv(sample_feature_data_path + '%s.csv' % (feat_1 + '_' + feat_2))
#         bs = BayesianSmoothing(1, 1)
#         bs.update(count[feat_1 + '_' + feat_2 + '_all'].values, count[feat_1 + '_' + feat_2 + '_1'].values, 1000, 0.001)
#         count[feat_1 + '_' + feat_2 + '_ctr'] = (count[feat_1 + '_' + feat_2 + '_1'] + bs.alpha) / (
#                 count[feat_1 + '_' + feat_2 + '_all'] + bs.alpha + bs.beta)
#         count[feat_1 + '_' + feat_2 + '_ctr'] = count[feat_1 + '_' + feat_2 + '_ctr'].apply(lambda x: float('%.6f' % x))

        # count.drop([feat_1 + '_' + feat_2 + '_1', feat_1 + '_' + feat_2 + '_all'], axis=1, inplace=True)
        count[feat_1 + '_' + feat_2 + '_ctr'] = 100*(count[feat_1 + '_' + feat_2 + '_1'] +0.0001) / (
            count[feat_1 + '_' + feat_2 + '_all'] + 0.0001)
        count[feat_1 + '_' + feat_2 + '_ctr'] = count[feat_1 + '_' + feat_2 + '_ctr'].apply(lambda x: float('%.4f' % x))
        count.drop([feat_1 + '_' + feat_2 + '_1', feat_1 + '_' + feat_2 + '_all'], axis=1, inplace=True)
        data = data.merge(count, how='left', on=[feat_1, feat_2])
#         data.fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)
        data.fillna(value=0, inplace=True)

        print("concat " + feat_1 +'_' + feat_2+ " over")

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
        
    output = open(raw_data_path+ "data_smooth.pkl", 'wb')    
    pickle.dump(data, output, protocol=4)
    output.close()
    del data
    gc.collect()


# In[10]:

gen_data()
gen_features()
add_AllFeatures(data)


# In[12]:


f = open(raw_data_path + 'data_smooth.pkl', 'rb')
train = pickle.load(f)
f.close()

print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('-----------------------read count feature data-----------')

# In[13]:


def func(x1, x2):
    return str(x1)+'_'+str(x2)

def func2(x1, x2, x3):
    return str(x1)+'_'+str(x2)+"_"+str(x3)


# In[14]:


def cross_feat(train, fea1, fea2):
    value = list(map(func, train[fea1], train[fea2]))
#     dummies = pd.get_dummies(value, prefix='{}_{}'.format(fea1, fea2))
    feat = pd.Series(value, name='{}_{}'.format(fea1, fea2))
    return feat

def cross_3feat(train, fea1, fea2, fea3):
    value = list(map(func2, train[fea1], train[fea2], train[fea3]))
    feat = pd.Series(value, name='{}_{}_{}'.format(fea1, fea2, fea3))
    return feat


# In[15]:


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
#cross_3fea = [('education', 'marriageStatus', 'productType'),('education', 'marriageStatus', 'carrier'),  ('education', 'age', 'carrier')]


# In[16]:


one_hot_feature=['age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']


# In[17]:


# for fea1, fea2, fea3 in cross_3fea:
#     fea1_fea2_fea3 = cross_3feat(train, fea1, fea2, fea3)
#     train[fea1_fea2_fea3.name] = fea1_fea2_fea3
#     one_hot_feature.append((fea1_fea2_fea3.name))
for fea1, fea2 in cross_2fea:
     fea1_fea2 = cross_feat(train, fea1, fea2)
     train[fea1_fea2.name] = fea1_fea2
     one_hot_feature.append(fea1_fea2.name)


# # In[18]:


data = train
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('-----------------------LabelEncoder feature data Down-----------')

output = open(raw_data_path+ "LabelEncode_data.pkl", 'wb')    
pickle.dump(data, output, protocol=4)
output.close()
print('-----------------------LabelEncoder feature data Saved-----------')



# In[19]:

#f = open(raw_data_path + 'LabelEncode_data.pkl', 'rb')
#data = pickle.load(f)
#f.close()

#train=data[data.label!=-1]
#train_y=train.pop('label')
#test=data[data.label==-1]

#train_y.to_csv('../data/train_y,csv', index=False)
#print('-----------------------train_y data saved-----------')

#train_encoded = pd.read_csv(raw_data_path + 'train_encoded.csv')
#test_encoded = pd.read_csv(raw_data_path + 'test_encoded.csv')
#train['LBS_enc'] = train_encoded
#test['LBS_enc'] = test_encoded
#print('-----------------------encode data concat-----------')

# In[20]:


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
    train.drop('LBS', axis=1, inplace=True)
    test['LBS_enc'] = test_encoded
    test.drop('LBS', axis=1, inplace=True)

    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('-----------------------target_encode feature data Down-----------')


#target_enc(train, test, train_y)

#print('-----------------------encode data concat-----------')

#output = open(raw_data_path+ "encode_data.pkl", 'wb')    
#pickle.dump(data, output, protocol=4)
#output.close()
#print('-----------------------target_encode data saved-----------')




# In[32]:

# In[35]:
def one_hot_gen(train, test):
    
    vector_feature= ['interest1','interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']

    fea_list = []
    for i in train.columns:
        if i not in one_hot_feature and i not in vector_feature and i not in ['aid', 'uid']: 
            fea_list.append(i) 


    # In[34]:

    print('-----------------------start One-hot encode------------------')
    enc = OneHotEncoder()
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
   




    # In[36]:

    cv=CountVectorizer(min_df=0.0009)
    for feature in vector_feature:
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('cv prepared !')
    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

  
    sparse.save_npz('../data/train_x.npz', train_x)
    sparse.save_npz('../data/test_x.npz', test_x)

    print('data saved !')
# In[42]:

#one_hot_gen(train, test)


print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
