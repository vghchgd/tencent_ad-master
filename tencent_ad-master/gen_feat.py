import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn import cross_validation, metrics
import scipy.special as special
from sklearn.externals import joblib
import gc, os
import datetime
import pickle


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
# train=pd.read_csv('../data/train.csv')
# predict=pd.read_csv('../data/test1.csv')
# train.loc[train['label']==-1,'label']=0
# predict['label']=-1
# data=pd.concat([train,predict])
# data=pd.merge(data,ad_feature,on='aid',how='left')
# data=pd.merge(data,user_feature,on='uid',how='left')

data = pd.read_csv('../data/sample_train.csv')

sample_feature_data_path = '../data/count_feat/'
raw_data_path = '../data/raw_data/'

drop_feature = ['appIdAction','appIdInstall','interest3','interest4','kw3','topic3']
data.drop(drop_feature, axis=1,inplace=True)

train_y = data.pop('label')
train = data

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)


def gen_features():

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



def add_AllFeatures():

    data = train
    count = pd.read_csv(sample_feature_data_path + 'uid_adCount.csv')
    data = data.merge(count, how='left', on='uid')
    data.fillna(value=0, inplace=True)
    print ('uid over')

    for feat_1 in ['creativeId', 'aid', 'advertiserId', 'campaignId', 'adCategoryId',
                   'education', 'LBS', 'consumptionAbility', 'marriageStatus', 'age', 'house']:
        count = pd.read_csv(sample_feature_data_path + '%s.csv' % (feat_1))
        bs = BayesianSmoothing(1, 1)
        bs.update(count[feat_1 + '_all'].values, count[feat_1 + '_1'].values, 1000, 0.001)
        count[feat_1 + '_ctr'] = (count[feat_1 + '_1'] + bs.alpha) / (
                count[feat_1 + '_all'] + bs.alpha + bs.beta)
        count[feat_1 + '_ctr'] = count[feat_1 + '_ctr'].apply(lambda x: float('%.6f' % x))
        # count.drop([feat_1 + '_1', feat_1 + '_all'], axis=1, inplace=True)
        data = data.merge(count, how='left', on=feat_1)
        data.fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)

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
        bs = BayesianSmoothing(1, 1)
        bs.update(count[feat_1 + '_' + feat_2 + '_all'].values, count[feat_1 + '_' + feat_2 + '_1'].values, 1000, 0.001)
        count[feat_1 + '_' + feat_2 + '_ctr'] = (count[feat_1 + '_' + feat_2 + '_1'] + bs.alpha) / (
                count[feat_1 + '_' + feat_2 + '_all'] + bs.alpha + bs.beta)
        count[feat_1 + '_' + feat_2 + '_ctr'] = count[feat_1 + '_' + feat_2 + '_ctr'].apply(lambda x: float('%.6f' % x))

        # count.drop([feat_1 + '_' + feat_2 + '_1', feat_1 + '_' + feat_2 + '_all'], axis=1, inplace=True)
        data = data.merge(count, how='left', on=[feat_1, feat_2])
        data.fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)

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
        
    output = open(raw_data_path+ "sample_data_smooth.pkl", 'wb')    
    pickle.dump(data, output, protocol=4)
    output.close()
    del data
    gc.collect()



gen_features()

add_AllFeatures()

print ('fea_count-------- DOWN---------------')

f = open(raw_data_path + 'sample_data_smooth.pkl', 'rb')
train = pickle.load(f)
f.close()

def func(x1, x2):
    return str(x1)+'_'+str(x2)

def func2(x1, x2, x3):
    return str(x1)+'_'+str(x2)+"_"+str(x3)


def cross_feat(train, fea1, fea2):
    value = list(map(func, train[fea1], train[fea2]))
    dummies = pd.get_dummies(value, prefix='{}_{}'.format(fea1, fea2))
    return dummies

def cross_3feat(train, fea1, fea2, fea3):
    value = list(map(func2, train[fea1], train[fea2], train[fea3]))
    dummies = pd.get_dummies(value, prefix='{}_{}_{}'.format(fea1, fea2, fea3))
    return dummies


cross_2fea_use = [('productType', 'creativeSize'),
 ('productType', 'marriageStatus'),
 ('productType', 'os'),
 ('age', 'education'),
 ('age', 'carrier'),
 ('education', 'marriageStatus'),
 ('education', 'carrier'),
 ('creativeSize', 'carrier'),
 ('consumptionAbility', 'carrier'),
 ('marriageStatus', 'os'),
 ('education', 'age', 'carrier')
 ]
cross_3fea_use = [('education', 'marriageStatus', 'productType'),('education', 'marriageStatus', 'carrier')]


for fea1, fea2 in cross_2fea_list:
    fea1_fea2 = cross_feat(train, fea1, fea2)
    train_df = sparse.hstack((fea1_fea2, load_data))

print ('fea1_fea2_cross-------- DOWN---------------')

for fea1, fea2, fea3 in cross_3fea_list:
    fea1_fea2_fea3 = cross_3feat(train, fea1, fea2, fea3)
    train_df = sparse.hstack((fea1_fea2_fea3, train_df))

print ('fea1_fea2_fea3_cross-------- DOWN---------------')


