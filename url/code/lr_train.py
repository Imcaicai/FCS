'''
用逻辑回归计算
'''


from scipy import interpolate
from tkinter import Y
import pandas as pd
import numpy as np
import sklearn
from warnings import simplefilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

# 忽略所有 future warnings
simplefilter(action='ignore', category=FutureWarning)

def LR():
    # 加载数据集
    data = pd.read_csv("data/feature_train.csv")

    # 提取特征值和目标值
    feature = data[['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname']]
    label = data['label']

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2)
    print("训练集：", x_train.shape, y_train.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 归一化
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)

    # 建立模型
    LR = LogisticRegression()
    
    # 计算accuracy、precision、recall
    LR.fit(x_train, y_train)
    y_LR = LR.predict(x_test)
    print('逻辑回归准确率: ', accuracy_score(y_test, y_LR))
    y_scores = LR.predict_proba(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores[:,1])
    
    # 通过sklear方法进行绘制 PR 曲线
    plt.plot(recall, precision, drawstyle="steps-post")
    plt.xlabel("Recall (Positive label: 1)")
    plt.ylabel("Precision (Positive label: 1)")
    plot_precision_recall_curve(LR, x_test, y_test)
    plt.savefig('img/f2.png')
    
    # 计算 score
    pr=np.array((recall, precision)).T
    pr = pr[pr[:,0].argsort()] #按照第1列排序
    pr_df = pd.DataFrame(pr, columns=['recall','precision'])
    pr_df = pr_df.drop_duplicates(subset='recall')
    # pr_data.to_csv('data/pr.csv', index=False)
    
    f=interpolate.interp1d(pr_df['recall'],pr_df['precision'],kind='quadratic')
    p=[0.7, 0.8, 0.9]
    r=f(p)
    score=0.5*r[0]+0.3*r[1]+0.2*r[2]
    print(score)

if __name__ == "__main__":
    LR()