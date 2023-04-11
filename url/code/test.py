'''
用随机森林预测
'''

from unittest import result
from scipy import interpolate
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from warnings import simplefilter

# 忽略所有 future warnings
simplefilter(action='ignore', category=FutureWarning)

'''
预测test.csv
train_path:     训练集路径
test_path:      测试集路径
result_path:    结果路径
'''
def predict(train_path,test_path,result_path):
    # 加载数据
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 构造特征值和目标值
    # 1不包括ip特征，2包括ip特征
    x_train_1 = train[['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname']]
    x_train_2 = train[['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname','addr','longitude','latitude']]
    y_train = train['label']
    url = test['url']
    x_test_1 = test[test.isnull().T.any()]
    url_1 = x_test_1['url']
    x_test_1 = x_test_1[['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname']]
    
    x_test_2 = test.dropna(axis=0,subset=['longitude'])
    url_2 = x_test_2['url']
    x_test_2 = x_test_2[['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname','addr','longitude','latitude']]

    # 建立模型
    RF_1 = RandomForestClassifier(n_estimators=100, max_depth=6, max_features=6)
    RF_2 = RandomForestClassifier(n_estimators=100, max_depth=8, max_features=6)

    # 训练
    RF_1.fit(x_train_1.values, y_train)
    RF_2.fit(x_train_2.values, y_train)

    # 预测
    label_1 = RF_1.predict(x_test_1.values)
    label_2 = RF_2.predict(x_test_2.values)
    
    print("url_1：", url_1.shape)
    print("url_2：", url_2.shape)
    
    # 合并结果
    result_1 = pd.DataFrame(np.array((url_1, label_1)).T, columns=['url','label_1'])
    result_2 = pd.DataFrame(np.array((url_2, label_2)).T, columns=['url','label_2'])
    
    result = pd.merge(url, result_1, on='url', how='left')
    print("result：", result.shape)
    result = pd.merge(result, result_2, on='url', how='left')
    print("result：", result.shape)
    
    result.to_csv(result_path, index=False)
    

    
if __name__ == "__main__":
    predict('data/train_feature.csv','data/test_feature.csv','data/result.csv')
