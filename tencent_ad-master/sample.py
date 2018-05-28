import pandas as pd
import math
import numpy as np
import  gc,os
import scipy.special as special
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
#from utils import raw_data_path, sample_feature_data_path, result_path, cache_pkl_path, dump_pickle, load_pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle


raw_path = '../data/raw_data/'
sample_path = '../data/sample_data/'
train = sparse.load_npz(raw_path + 'train_x_20_ctrtop38.npz')
train_x = train.tocsr()
y = pd.read_csv('../data/train_y_518.csv', header=None)
print(train_x.shape, y.shape)
train, test, train_y, test_y = train_test_split(train_x, y, test_size=0.2, random_state=2021)
output = open(sample_path+ "train_x_sample.pkl", 'wb')
pickle.dump(test, output, protocol=4)
output.close()
test_y.to_csv(sample_path + 'train_y_sample.csv', index=False)
