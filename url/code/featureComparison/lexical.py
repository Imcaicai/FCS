'''
Lexical Feature set
1. 分隔符总数                   都差不多                  去掉准确率提高一点点(0.003左右)
2. 连字符总数                   都差不多                  去掉对准确率提高一点(0.01左右)
3. 主机名的长度                 正常的一般比不正常的多      去掉对准确率影响很大
4. 点数                        都差不多                   去掉准确率提高一点(0.01左右)
5. 二进制特征-顶级域                                      去掉准确率提高一点(0.01左右)(但可能是域名没有选好)
6. 二进制特征-可疑单词                                    去掉准确率下降一点(0.01左右)(还需要好好提炼)
'''


import pandas as pd
import numpy as np
import tldextract

# 正常顶级域
normal_tld = ['com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'co_jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv', 'me', 'name', 'net', 'nz', 'org', 'us']
# 可疑单词
suspicious_word = ['update', 'click', 'www.', 'link']
# 可疑顶级域
suspicious_tld=['zip','cricket','link','work','party','gq','kim','country','science','tk']

dataset_path = '../data/train_dataset.csv'
lexical_feature_path = 'lexical_feature_set.npy'


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



if __name__ == '__main__':

    # 读取数据集
    df = pd.read_csv(dataset_path, header=0)

    # 获取 lexical feature set
    url = df['url']
    deli_num = []
    hyp_num = []
    url_len = []
    dot_num = []
    nor_tld_token = []
    sus_word_token = []

    for i in url:
        deli_num.append(get_deli_num(i))
        hyp_num.append(get_hyp_num(i))
        url_len.append(get_url_len(i))
        # 点数去掉，准确率会提高一点
        # dot_num.append(get_dot_num(i))
        nor_tld_token.append(is_normal_tld(i))
        sus_word_token.append(is_suspicious_word(i))


    # 形成 lexical feature set
    lexical_feature = np.array((deli_num,hyp_num,url_len,nor_tld_token, sus_word_token)).T


    # 保存 lexical feature set
    np.save('lexical_feature_set.npy', lexical_feature)
    basic = pd.DataFrame(lexical_feature, columns=['deli_num','hyp_num','url_len','nor_tld_token', 'sus_word_token'])
    basic.to_csv('lexical_feature_set.csv', index=False)
