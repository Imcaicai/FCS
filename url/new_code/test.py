import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.preprocessing import MinMaxScaler
import joblib

# 忽略所有 future warnings
simplefilter(action='ignore', category=FutureWarning)

# 设置路径
test_path = 'data/test.csv'
result_path = 'data/result.csv'
abnormal_result_path = 'data/abnormal_result.csv'
final_result_path = 'data/final_result.csv'
lex_test_path = 'featureComparison/lexical_test.csv'

def fit(test_set):
    x_test = test_set
    x_test = MinMaxScaler().fit_transform(x_test)

    LR = joblib.load('./saved_model/LR.pkl')
    y_test = LR.predict(x_test)

    result = pd.DataFrame(y_test, columns=['label'])
    result.to_csv(result_path, index=False)

    x_test_abnormal = []
    y_test_final = y_test
    LR_abnormal = joblib.load('./saved_model/LR_abnormal.pkl')
    for i in range(x_test.shape[0]):
        if y_test[i] == 1:
            x_test_abnormal.append(x_test[i])
    y_test_abnormal = LR_abnormal.predict(x_test_abnormal)
    result = pd.DataFrame(y_test_abnormal, columns=['label'])
    result.to_csv(abnormal_result_path, index=False)

    cnt = 0
    for i in range(x_test.shape[0]):
        if y_test[i] == 1:
            y_test_final[i] = y_test_abnormal[cnt]
            cnt += 1
    result = pd.DataFrame(y_test_final, columns=['label'])
    result.to_csv(final_result_path, index=False)


lex_test_feature = pd.read_csv(lex_test_path,
                               usecols=['deli_num', 'hyp_num', 'url_len', 'nor_tld_token', 'sus_word_token',
                                        'ip_in_hostname'])

df_test = pd.read_csv(test_path, header=0)
fit(lex_test_feature)
