对需要用到的数据进行了一定的拆分

abnormal.csv是所有不正常的url

abnormal_result.csv是result.csv中label=1的url的具体类别

final_result.csv是测试集所有url的具体类别

normal.csv是正常的url

result.csv是对测试集进行正常与不正常的二分类后的结果

test.csv是测试集

x_train.csv是用来训练的url

x_train_abnormal.csv是x_train的url中不正常的url

y_train.csv是x_train的每个url对应的label（二分类）

y_train_abnormal.csv是x_train的url中不正常的url的具体类别

