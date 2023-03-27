'''
用随机森林训练
'''

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

def forest():
    # 加载数据
    data = pd.read_csv("data/feature_train.csv")

    # 构造特征值和目标值
    feature = data[['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname']]
    label = data['label']

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2)
    print("训练集：", x_train.shape, y_train.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 建立模型
    RF = RandomForestClassifier(n_estimators=100, max_depth=8, max_features=5)

    # # 超参数搜索
    # param = {"n_estimators":[50, 55, 60, 65, 70], "max_depth":[4, 6, 8, 10, 12]}
    # gc = GridSearchCV(rf, param_grid=param, cv=5)
    # print("在验证集上的准确率：", gc.best_score_)
    # print("最好的模型参数：", gc.best_params_)
    # print("最好的模型：", gc.best_estimator_)

    # 训练
    RF.fit(x_train, y_train)

    # 计算准确率
    y_RF = RF.predict(x_test)
    print('随机森林准确率: ', accuracy_score(y_test, y_RF))
    y_scores = RF.predict_proba(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores[:,1])
    
    # 通过sklear方法进行绘制 PR 曲线
    plt.plot(recall, precision, drawstyle="steps-post")
    plt.xlabel("Recall (Positive label: 1)")
    plt.ylabel("Precision (Positive label: 1)")
    plot_precision_recall_curve(RF, x_test, y_test)
    plt.savefig('img/f1.png')
    
    # 计算 score
    pr=np.array((recall, precision)).T
    pr = pr[pr[:,0].argsort()] #按照第1列排序
    pr_df = pd.DataFrame(pr, columns=['recall','precision'])
    pr_df = pr_df.drop_duplicates(subset='recall')
    f=interpolate.interp1d(pr_df['recall'],pr_df['precision'],kind='quadratic')
    p=[0.7, 0.8, 0.9]
    r=f(p)
    score=0.5*r[0]+0.3*r[1]+0.2*r[2]
    print(score)
    


if __name__ == "__main__":
    forest()
