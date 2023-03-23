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

# 设置路径
train_path = 'data/train.csv'
test_path = 'data/test.csv'
result_path = 'data/result.csv'

lex_train_path = 'featureComparison/lexical_train.csv'
lex_test_path = 'featureComparison/lexical_test.csv'

# 读取数据集
df_train = pd.read_csv(train_path, header=0)
df_test = pd.read_csv(test_path, header=0)
label = df_train['label'].values



# feature selection
# 2. lexical feature
lex_train_fea = pd.read_csv(lex_train_path,usecols=['deli_num','hyp_num','url_len','nor_tld_token','sus_word_token'])
lex_test_fea = pd.read_csv(lex_test_path,usecols=['deli_num','hyp_num','url_len','nor_tld_token','sus_word_token'])


'''
用不同分类器分类
'''
def classification(train_set, test_set, label, flag):

    if flag==0: # 用于训练，输出准确率
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(train_set.astype(float), label, test_size=0.2, random_state=123)

        # 归一化
        x_train = MinMaxScaler().fit_transform(x_train)
        x_test = MinMaxScaler().fit_transform(x_test)

        # 分类器算法
        LR = LogisticRegression(penalty='l2')

        # 准确率
        LR.fit(x_train, y_train)
        # y_LR = LR.predict(x_test)
        # print('Logistic Regression prediction: ', accuracy_score(y_test, y_LR))
        y_scores = LR.predict_proba(x_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores[:,1])
        
        # 通过sklear方法进行绘制 PR 曲线
        plt.plot(recall, precision, drawstyle="steps-post")
        plt.xlabel("Recall (Positive label: 1)")
        plt.ylabel("Precision (Positive label: 1)")
        plot_precision_recall_curve(LR, x_test, y_test)
        plt.show()
        
        return 0

    # 用于输出预测结果
    x_train = train_set
    y_train = label
    x_test = test_set

    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)
    LR = LogisticRegression(penalty='l2')

    y_test = LR.fit(x_train, y_train).predict(x_test)
    # 保存 lexical feature set
    result = pd.DataFrame(y_test, columns=['label'])
    result.to_csv(result_path, index=False)

    return 0


print('----------------------------------------------------')
print('2. Lexica Feature Set Classification Results')
classification(lex_train_fea,lex_test_fea,label,0)
print('\n')
