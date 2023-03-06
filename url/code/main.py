from tkinter import Y
import pandas as pd
import numpy as np
import sklearn
from warnings import simplefilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler


# 忽略所有 future warnings
simplefilter(action='ignore', category=FutureWarning)

# 设置路径
basic_feature_path = 'featureComparison/basic_feature_set.npy'
lexical_feature_path = 'featureComparison/lexical_feature_set.npy'
dataset_path = 'data/train_dataset_4.csv'

# 读取数据集
df = pd.read_csv(dataset_path, header=0)
label = df['label'].values


# feature selection
# 1. basic feature
basic_feature = np.load(basic_feature_path).astype(int)
df_basic = pd.DataFrame(data=basic_feature, columns=['ip_in_hostname', 'url_len'])

# 2. lexical feature
lexical_feature = np.load(lexical_feature_path).astype(int)
df_lexical = pd.DataFrame(data=lexical_feature, columns=['deli_num','hyp_num','url_len','nor_tld_token','sus_word_token'])



'''
用不同分类器分类
'''
def classification(feature_set, label):

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(feature_set.astype(float), label, test_size=0.2, random_state=123)
    
    # 归一化
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)

    # 分类器算法
    LR = LogisticRegression(penalty='l1')

    # 交叉检验
    print('LR cross value: ', cross_val_score(LR, x_train, y_train, cv=5, scoring='accuracy').mean())

    # 预测
    y_LR = LR.fit(x_train, y_train).predict(x_test)
    # y_LR_proba = LR.fit(x_train, y_train).predict_proba(x_test)
    # print(y_LR_proba)
    print('Logistic Regression prediction: ', accuracy_score(y_test, y_LR))

    return 0

print('----------------------------------------------------')
print('1. Basic Feature Classification Results')
classification(basic_feature,label)
print('\n')

print('----------------------------------------------------')
print('2. Lexica Feature Set Classification Results')
classification(lexical_feature,label)
print('\n')