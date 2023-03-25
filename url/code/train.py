import csv
from random import randrange, seed
from tkinter import Y
import pandas as pd
import numpy as np
import sklearn
from warnings import simplefilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

# 忽略所有 future warnings
simplefilter(action='ignore', category=FutureWarning)

'''
加载数据
'''
def load_csv(filename):
    data_set=[]
    with open(filename,'r') as file:
        csv_reader=csv.reader(file)
        for line in csv_reader:
            data_set.append(line)
    return data_set
 
'''
除了判别列，其他列都转换为float类型
'''
def column_to_float(data_set):
    feat_len=len(data_set[0])-1
    for data in data_set:
        for column in range(feat_len):
            data[column]=float(data[column].strip())    # strip()返回移除字符串头尾指定的字符生成的新字符串。

'''
将数据集分成N份，每份包含 fold_size 个值
每个值由 data_set 的内容随机产生，每个值被用 1 次
'''
def spiltdata_set(data_set,n):
    fold_size=int(len(data_set)/n)      
    data_set_copy=list(data_set)    # 复制一份data_set,防止内容改变
    data_set_spilt=[]
    for i in range(n):
        fold=[]
        while len(fold) < fold_size:   
            index=randrange(len(data_set_copy))
            fold.append(data_set_copy.pop(index))   # 将对应索引index的内容从data_set_copy中导出并删除
        data_set_spilt.append(fold)
    return data_set_spilt   # 由data_set分出的n块数据构成的列表

'''
构造数据集的随机子样本
'''
def get_subsample(data_set,ratio):
    sub_data_set=[]
    lenSubdata=round(len(data_set)*ratio)    # round()方法返回浮点数x的四舍五入值
    while len(sub_data_set) < lenSubdata:
        index=randrange(len(data_set)-1)
        sub_data_set.append(data_set[index])    # 有放回的随机采样
    return sub_data_set
 
'''
根据特征和特征值分割数据集
'''
def data_spilt(data_set,index,value):
    left=[]
    right=[]
    for row in data_set:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left,right

'''
计算分割代价
分类越准确，则 loss 越小
'''
def spilt_loss(left,right,class_values):
    loss=0.0
    for class_value in class_values:
        left_size=len(left)
        if left_size!=0:  #防止除数为零
            prop=[row[-1] for row in left].count(class_value)/float(left_size)
            loss += (prop*(1.0-prop))
        right_size=len(right)
        if right_size!=0:
            prop=[row[-1] for row in right].count(class_value)/float(right_size)
            loss += (prop*(1.0-prop))
    return loss

'''
选取任意的n个特征，在这n个特征中，选取分割时的最优特征
'''
def get_best_spilt(data_set,n_features):
    features=[]
    label=list(set(row[-1] for row in data_set))    # 标签值
    b_index,b_value,b_loss,b_left,b_right=999,999,999,None,None
    while len(features) < n_features:
        index=randrange(len(data_set[0])-1) # 往features添加n_features个特征，特征索引随机取
        if index not in features:
            features.append(index)
    for index in features:  # 在n_features个特征中选出最优的特征索引，保证每课决策树的差异性
        for row in data_set:
            left,right=data_spilt(data_set,index,row[index])
            loss=spilt_loss(left,right,label)
            if loss < b_loss:
                # 最后得到最优的分类特征b_index,分类特征值b_value,分错的代价成本b_loss,分类结果b_left b_right
                b_index,b_value,b_loss,b_left,b_right=index,row[index],loss,left,right
    #print b_loss
    return {'index':b_index,'value':b_value,'left':b_left,'right':b_right}

'''
决定输出标签
'''
def decide_label(data):
    output=[row[-1] for row in data]
    return max(set(output),key=output.count)

'''
子分割，不断地构建叶节点的过程
'''
def sub_spilt(root,n_features,max_depth,min_size,depth):
    left=root['left']
    #print left
    right=root['right']
    del(root['left'])
    del(root['right'])
    #print depth
    if not left or not right:
        root['left']=root['right']=decide_label(left+right)
        #print 'testing'
        return
    if depth > max_depth:
        root['left']=decide_label(left)
        root['right']=decide_label(right)
        return
    if len(left) < min_size:
        root['left']=decide_label(left)
    else:
        root['left'] = get_best_spilt(left,n_features)
        #print 'testing_left'
        sub_spilt(root['left'],n_features,max_depth,min_size,depth+1)
    if len(right) < min_size:
        root['right']=decide_label(right)
    else:
        root['right'] = get_best_spilt(right,n_features)
        #print 'testing_right'
        sub_spilt(root['right'],n_features,max_depth,min_size,depth+1)  

'''
构造决策树
'''
def build_tree(data_set,n_features,max_depth,min_size):
    root=get_best_spilt(data_set,n_features)
    sub_spilt(root,n_features,max_depth,min_size,1) 
    return root

def bagging_predict(trees,row):
    predictions=[predict(tree,row) for tree in trees]
    return max(set(predictions),key=predictions.count)

'''
创建随机森林
'''
def random_forest(train,test,ratio,n_feature,max_depth,min_size,n_trees):
    trees=[]
    for i in range(n_trees):    # n_trees表示决策树的数量
        sub_train=get_subsample(train,ratio)    # 随机采样保证了每棵决策树训练集的差异性
        tree=build_tree(sub_train,n_feature,max_depth,min_size)
        trees.append(tree)  # 建立一个决策树
    #predict_values = [predict(trees,row) for row in test]
    predict_values = [bagging_predict(trees, row) for row in test]
    return predict_values

'''
预测测试集结果
'''
def predict(tree,row):
    predictions=[]
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'],dict):
            return predict(tree['left'],row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'],dict):
            return predict(tree['right'],row)
        else:
            return tree['right']

'''
计算准确率
'''
def accuracy(predict_values,actual):
    correct=0
    for i in range(len(actual)):
        if actual[i]==predict_values[i]:
            correct+=1
    return correct/float(len(actual))  



if __name__=='__main__':
    seed(1)
    data_set=load_csv('data/feature_train.csv')    # 读取数据集：特征值+标签(最后一行)
    column_to_float(data_set)   # 将值转为 float

    # 参数调优
    n_folds=5
    max_depth=15    # 树深
    min_size=1
    ratio=1.0
    n_features=15   # 生成单棵决策树时的最大特征数
    n_trees=10  # 决策树的棵树

    folds=spiltdata_set(data_set,n_folds)   # 将数据集分成 n_flods份
    scores=[]
    for fold in folds:
        train_set=folds[:]  # 当train_set的值改变的时候，防止folds的值也改变
        train_set.remove(fold)
        #print len(folds)
        train_set=sum(train_set,[])  #将多个fold列表组合成一个train_set列表
        #print len(train_set)
        test_set=[]
        for row in fold:
            row_copy=list(row)
            row_copy[-1]=None
            test_set.append(row_copy)
        #for row in test_set:
           # print row[-1]
        actual=[row[-1] for row in fold]
        predict_values=random_forest(train_set,test_set,ratio,n_features,max_depth,min_size,n_trees)
        accur=accuracy(predict_values,actual)   # 每份的准确率
        scores.append(accur)
    print ('Trees is %d'% n_trees)
    print ('scores:%s'% scores)
    print ('mean score:%s'% (sum(scores)/float(len(scores))))   # 平均准确率