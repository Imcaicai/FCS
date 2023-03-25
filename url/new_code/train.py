import pandas as pd
import numpy as np
from warnings import simplefilter
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# 忽略所有 future warnings
simplefilter(action='ignore', category=FutureWarning)

# 设置路径
lexical_train_path = 'featureComparison/lexical_train.npy'
lexical_train_abnormal_path = 'featureComparison/lexical_train_abnormal.npy'
dataset_path = 'data/y_train.csv'
abnormal_dataset_path = 'data/y_train_abnormal.csv'

# 读取数据集
df = pd.read_csv(dataset_path, header=0)
label = df['label'].values

df = pd.read_csv(abnormal_dataset_path, header=0)
abnormal_label = df['label'].values

def classification(feature_set, label, model_path):
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(feature_set.astype(float), label, test_size=0.2,
                                                        random_state=123)

    # 归一化
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)

    # 分类器算法
    LR = LogisticRegression(penalty='l2', max_iter=5000)

    # 交叉检验
    print('LR cross value: ', cross_val_score(LR, x_train, y_train, cv=5, scoring='accuracy').mean())

    # 预测
    y_LR = LR.fit(x_train, y_train).predict(x_test)
    print('Logistic Regression prediction: ', accuracy_score(y_test, y_LR))

    if max(y_train) == 1:
        y_scores = LR.predict_proba(x_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores[:, 1])

        # 通过sklearn方法进行绘制 PR 曲线
        plt.plot(recall, precision, drawstyle="steps-post")
        plt.xlabel("Recall (Positive label: 1)")
        plt.ylabel("Precision (Positive label: 1)")
        plot_precision_recall_curve(LR, x_test, y_test)
        plt.show()

    # save model
    joblib.dump(LR, model_path)


# lexical feature
lexical_train_feature = np.load(lexical_train_path).astype(int)
df_lexical = pd.DataFrame(data=lexical_train_feature,
                          columns=['deli_num', 'hyp_num', 'url_len', 'nor_tld_token', 'sus_word_token',
                                   'ip_in_hostname'])

lexical_train_abnormal_feature = np.load(lexical_train_abnormal_path).astype(int)
df_lexical_abnormal = pd.DataFrame(data=lexical_train_abnormal_feature,
                                   columns=['deli_num', 'hyp_num', 'url_len', 'nor_tld_token', 'sus_word_token',
                                            'ip_in_hostname'])
classification(lexical_train_feature, label, './saved_model/LR.pkl')
classification(lexical_train_abnormal_feature, abnormal_label, './saved_model/LR_abnormal.pkl')
