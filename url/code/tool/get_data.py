'''
正常的url太多了，这里用于随机提取部分正常的
'''

import pandas as pd
import numpy as np

# df.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
# n = number of rows(optional, cannot be used with frac) 抽取的行数；
# frac = fraction/proportion(optional, cannot be used with n) 抽取的比例；
# replace = Allow or disallow sampling of the same row more than once (boolean, default False) 是否为有放回抽样；
# weights (str or ndarray-like, optional) 权重
# random_state (int to use as interval, or call np.random.get_state(), optional) 整数作为间隔，或者调用np.random.get_state()
# axis = extract row or column (0->row, 1->column) 抽取行还是列（0是行，1是列）

combined=pd.read_csv('data/train_1.csv')
# random select 10% from dataset
sample = combined.sample(frac=0.3, random_state=5, axis=0)
# export to csv file
sample.to_csv('data/train_2.csv')

