【数据说明】
test.csv        		    题目给的所有测试集
result.csv      		    测试集test.csv的训练结果（标签为0和1）
abnormal.csv    		    题目给的所有训练集、不正常的网址（标签都设为1）
abnormal0.csv    		    题目给的所有训练集、不正常的网址（标签为1-12）
train.csv       		    全部的abnormal.csv + 抽样的正常url（训练集）
normal.csv      		    train1中所有 url 的标签及有效性
train_ip_feature.csv        train.csv的ip特征
test_ip_feature.csv         test.csv的ip特征
train_lexical_feature.csv   train.csv的lexical特征
train_featue.csv            train.csv的所有特征（过滤掉没有ip特征的样本）