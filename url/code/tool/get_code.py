'''
���ڸ�ѵ��������������ip��ַ��������
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


'''
��ѵ���������Լ���ip��ַ��������
train_path: ѵ������ip������ַ
test_path:  ���Լ���ip������ַ
'''
def addr2cdoe(train_path,test_path):
    train = pd.read_csv(train_path)
    test= pd.read_csv(test_path)
    
    # ɾ��ȱʧ����
    train_1 = train.dropna(axis=0,subset=['longitude'])
    test_1 = test.dropna(axis=0,subset=['longitude'])
    data = pd.concat([train_1,test_1])
    data=data[['url','addr','longitude','latitude']]

    # ��ip feature�Ĺ��ұ���
    data[['addr']] = LabelEncoder().fit_transform(data[['addr']])

    # �ֱ����ѵ���������Լ���ip����
    train_2 = pd.merge(train, data, on=['url'], how='left')
    # train_2 = train_2[['url','addr','longitude','latitude']]
    train_2.to_csv(train_path)
    test_2 = pd.merge(test, data, on=['url'], how='left')
    # test_2 = test_2[['url','addr','longitude','latitude']]
    test_2.to_csv(test_path)


if __name__ == '__main__':
    addr2cdoe('data/train_ip_feature.csv','data/test_ip_feature.csv')