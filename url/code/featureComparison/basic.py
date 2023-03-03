'''
Basic Feature set
1. 域名中的点数         正常的一般比不正常的多  去掉对准确率几乎没有影响
2. 域名中是否有IP地址   不正常的没有可能性更大   去掉对准确率有一点影响
3. 整个域名长度         正常的一般比不正常的长  去掉对准确率有很大影响
'''


from operator import index
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from sqlalchemy import false
import whois

'''
pd.set_option('display.max_columns', 10000) # 最多显示10000列
pd.set_option('display.max_rows', 10000)    # 最多显示10000行
pd.set_option('display.max_colwidth', 10000)    # 设置value的显示长度为10000
pd.set_option('display.width',1000) # 设置显示宽度
np.set_printoptions(threshold=np.inf)   # 输出数组时，不用省略号
'''


TLD = ['com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'co_jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv', 'me', 'name', 'net', 'nz', 'org', 'us']
dataset_path = '../data/train_dataset.csv'
basic_feature_path = 'basic_feature_set.npy'

'''
获取整个URL中的点数
'''
def get_dot_num(url):
    return url.count('.')

'''
获取整个URL长度
'''
def get_url_len(url):
    return len(url)

'''
主机名是否为IP地址
有则返回1，没有则返回0
'''
def is_ip_in_hostname(url):
    for i in url:
        if i.isdigit()==False and i!='.' and i!=':':
            return 1
    return 0



if __name__ == '__main__':
    # 1. 读取数据集
    df = pd.read_csv(dataset_path, header=0)
    
    # 2. 获取 Basic Feature
    url = df['url']
    label = df['label']
    dot_num = []
    ip_in_hostname = []
    url_len = []
    # creation_date = []
    # is_creation_date_value = []

    for i in url:
        # 点数影响并不大
        # dot_num.append(get_dot_num(i))
        ip_in_hostname.append(is_ip_in_hostname(i))
        url_len.append(get_url_len(i))
    
    # 3. 形成 Basic Feature set
    basic_feature = np.array((ip_in_hostname, url_len)).T

    # 4. 保存 Basic Feature set
    np.save('basic_feature_set.npy', basic_feature)
    basic = pd.DataFrame(basic_feature, columns=['ip_in_hostname', 'url_len'])
    basic.to_csv('basic_feature_set.csv', index=False)

