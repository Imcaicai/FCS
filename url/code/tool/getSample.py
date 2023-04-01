'''
用于抽取二分类训练集中不正常url的样本
'''
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin


"""
随机抽样: 每个样本等概率抽样
array: 待采样数组
size: 采样个数
replace: 是否放回，True为有放回的抽样，False为无放回的抽样
"""
def random_sample(array, size: int, replace=True):
    return np.random.choice(array, size=size, replace=replace)


"""聚类抽样: 也称整群抽样，先对样本聚出多个类，然后随机的抽类，抽中哪个类，这一类的所有样本点都会被抽出来，不会对单个点进行抽样
array: 样本点
"""
def cluster_sample(array):
    from sklearn.cluster import DBSCAN
    label = DBSCAN(eps=30, min_samples=3).fit(array).labels_  # 使用DBSCAN做聚类，这个可以换
    select_cluster = random_sample(np.unique(label), 1)  # 随机选择一个类
    return array[label == select_cluster]


"""系统抽样: 以固定的节奏从总体中抽样，隔step个抽1个，再隔step个抽一个，循环下去
array: 样本点
step: 步长
"""
def systematic_sample(array, step):
    select_index = list(range(0, len(array), 3))
    return array[select_index]


"""分层抽样: 先按照容量，给每个样本一些指标，然后样本内等概率抽样
array: 样本数据
label: 样本类别
size: 采样个数
"""
def stratify_sample(array, label, size: int):
    stratified_sample, _ = train_test_split(array, train_size=size, stratify=label)
    return stratified_sample


# def main():
#     # 构造数据
#     array_data = np.arange(0, 100)  # 待取样数据
#     array_label = np.random.random_integers(0, 5, size=100)  # 类别
#     # 开始采样
#     result = stratify_sample(array_data, array_label, 80)  # 分层抽样
#     print(array_data)
#     print(array_label)
#     print(result)


'''
过滤掉 normal 中标签不为 0 或无法打开的 url
结果保存在 normal_1 中
'''
def filter():
    data = np.array(pd.read_csv("data/normal.csv"))
    # url=data[:,0]
    # label=data[:,1]
    # is_open=data[:,2]
    result=[]
    for i in data:
        if i[1]==0 and i[2]==1:
            result.append(i[0])
    result_set = pd.DataFrame(result, columns=['url'])
    result_set.to_csv('data/normal_1.csv', index=False)


if __name__ == '__main__':
    # filter()

    # 加载数据集
    data = pd.read_csv("data/feature_normal_1.csv")
    feature = np.array(data[['deli_num','url_len']])
    all=np.array(data)
    
    # 画图
    fig = plt.figure()
    # fig.subplots_adjust()
    colors = ["#4EACC5", "#FF9C34", "#4E9A06"]
    
    # 构造一个聚类器
    n_clusters = 3
    k_means = KMeans(n_clusters=3)
    k_means.fit(feature) # 聚类
    k_means_cluster_centers = k_means.cluster_centers_  # 聚类中心
    # label_pred = k_means.labels_  # 获取聚类标签
    label_pred = pairwise_distances_argmin(feature, k_means_cluster_centers)    # 获取聚类标签

    # 可视化
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = label_pred == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(feature[my_members, 0], feature[my_members, 1], "w", markerfacecolor=col, marker=".")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.savefig('img/f3.png') # 保存图片
    plt.show()
    

    # 分层抽样
    result = stratify_sample(all, label_pred, 80000)  
    
    # 保存结果样本
    sample = np.array((result[:,0],result[:,1],result[:,2],result[:,3],result[:,4],result[:,5],result[:,6])).T
    sample_set1 = pd.DataFrame(sample, columns=['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname'])
    sample_set1.to_csv('data/feature_normal_2.csv', index=False)