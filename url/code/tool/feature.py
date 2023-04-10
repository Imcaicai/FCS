'''
All Feature set
1. 分隔符总数                   都差不多                  去掉准确率提高一点点(0.003左右)
2. 连字符总数                   都差不多                  去掉对准确率提高一点(0.01左右)
3. 主机名的长度                 正常的一般比不正常的多      去掉对准确率影响很大
4. 点数                        都差不多                   去掉准确率提高一点(0.01左右)
5. 二进制特征-顶级域                                      去掉准确率提高一点(0.01左右)(但可能是域名没有选好)
6. 二进制特征-可疑单词                                    去掉准确率下降一点(0.01左右)(还需要好好提炼)
7. 域名中是否有IP地址           不正常的没有可能性更大      去掉对准确率有一点影响
'''


import pandas as pd
import numpy as np
import tldextract
from sklearn.preprocessing import LabelEncoder

# 正常顶级域
normal_tld = ['com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'co_jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv', 'me', 'name', 'net', 'nz', 'org', 'us']
# 可疑单词
suspicious_word = ['update', 'click', 'www.', 'link']
# 可疑顶级域
suspicious_tld=['zip','cricket','link','work','party','gq','kim','country','science','tk']


'''
计算域名中分隔符的数量
'''
def get_deli_num(url):
    deli = ['-', '_', '?', '=', '&']
    count = 0
    for i in url:
        for j in deli:
            if i == j:
                count += 1
    return count


'''
计算域名中 '-' 的个数
'''
def get_hyp_num(url):
    return url.count('-')


'''
获取整个域名长度
'''
def get_url_len(url):
    return len(url)


'''
获取域名中的点数
'''
def get_dot_num(url):
    return url.count('.')


'''
域名的二进制特征
url 的顶级域是否为常见顶级域
'''
def is_normal_tld(url):

    tld = tldextract.extract(url).suffix
    if tld in normal_tld:
        return 0
    else:
        return 1


'''
域名的二进制特征
域名（去掉顶级域后）中是否含有可疑单词
'''
def is_suspicious_word(url):
    for i in suspicious_word:
        if i in url[:url.rfind('.')]:
            return 1
    return 0


'''
主机名是否为IP地址
有则返回1，没有则返回0
'''
def is_ip_in_hostname(url):
    for i in url:
        if i.isdigit()==False and i!='.' and i!=':':
            return 1
    return 0


'''
获取lexical feature
dataset_path:   读取的url路径
feature_path:   lexical feature的保存路径
'''
def get_feature(dataset_path,feature_path):
    # 读取数据集
    df = pd.read_csv(dataset_path, header=0)

    # 获取 lexical feature set
    url = df['url']
    label = df['label']
    deli_num = []   # 分隔符总数
    hyp_num = []    # 连字符总数
    url_len = []    # 主机名长度
    dot_num = []    # 点数
    nor_tld_token = []  # 二进制特征：顶级域
    sus_word_token = [] # 二进制特征：可疑单词
    ip_in_hostname = [] # 域名中是否有IP地址

    for i in url:
        deli_num.append(get_deli_num(i))
        hyp_num.append(get_hyp_num(i))
        url_len.append(get_url_len(i))
        dot_num.append(get_dot_num(i))
        nor_tld_token.append(is_normal_tld(i))
        sus_word_token.append(is_suspicious_word(i))
        ip_in_hostname.append(is_ip_in_hostname(i))


    # 形成 lexical feature set
    feature = np.array((deli_num,hyp_num,url_len,dot_num,nor_tld_token, sus_word_token,ip_in_hostname,url,label)).T

    # 保存 lexical feature set
    feature_set = pd.DataFrame(feature, columns=['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname','url','label'])
    # feature_set = pd.DataFrame(feature)
    feature_set.to_csv(feature_path, index=False)


'''
合并lexical feature和ip feature
删除ip feature缺失的行
再次对normal的url抽样，使不正常url:正常url=1:3
lexical_path:   lexical feature的读取路径 
ip_path:        ip feature的读取路径
feature_path:   合并过滤后feature的保存路径
'''
def merge_feature(lexical_path,ip_path,feature_path):
    # 读取数据集
    lexical = pd.read_csv(lexical_path)
    ip= pd.read_csv(ip_path)
    
    # 删除ip feature缺失的行
    ip = ip.dropna(axis=0,subset=['longitude'])

    # 给ip feature的国家编码
    ip[['addr']] = LabelEncoder().fit_transform(ip[['addr']])

    # 合并lexical feature和ip feature
    data = pd.merge(lexical, ip, on=['url'], how='right')
    data = data[['deli_num','hyp_num','url_len','dot_num','nor_tld_token', 'sus_word_token','ip_in_hostname','url','label','addr','longitude','latitude']]
    
    # 控制不正常url:正常url=3:1
    abnormal=data.head(17000)               # 正常url：前17000条
    normal=data.tail(data.shape[0]-17000)   # 正常url：后面的data.shape[0]-17000条
    normal=normal.sample(n=17000*3)         # 取样，保证1:3
    result = pd.concat([abnormal,normal])   # 合并不正常的和正常的
    result.to_csv(feature_path)

if __name__ == '__main__':
    # # 生成lexical feature
    # dataset_path = 'data/normal_1.csv'
    # feature_path = 'data/feature_normal_1.csv'
    # get_feature(dataset_path,feature_path)

    # 合并lexical feature和ip feature
    merge_feature('data/train_lexical_feature.csv','data/train_ip_feature.csv','data/train_feature.csv')