'''
提取URL中的域名
由于大部分数据只有域名，因此分类时只考虑域名，不考虑路径
'''

from mimetypes import guess_all_extensions
from xml import dom
import pandas as pd
import numpy as np
from tld import get_tld

def get_domain(url):
    domain = ''
    for i in url:
        if i == '/':
            return domain
        else:
            domain = domain + i
            
    return domain 


url_path = 'data/abnormal.csv'
df = pd.read_csv(url_path, header=0)
url = df['url']
label = df['label']
domain_path = []
domain = []

# 去掉域名前面的http:// 或 https://
for i in url:
    if i[0:7] == 'http://':
        if i[-1] == '/':
            domain_path.append(i[7:-1])
        else:
            domain_path.append(i[7:])
    elif i[0:8] == 'https://':
        if i[-1] == '/':
            domain_path.append(i[8:-1])
        else:
            domain_path.append(i[8:])
    else:
        domain_path.append(i)


df_domain_path = pd.DataFrame({'url':domain_path})
df_domain_path.to_csv('domain_path.csv', index=False,sep=',')


# 去掉域名后面的路径
for i in domain_path:
    domain.append(get_domain(i))
df_domain = pd.DataFrame({'url':domain})
df_domain.to_csv('domain.csv', index=False,sep=',')

domain_label = np.array((domain, label.T)).T
df_domain = pd.DataFrame(domain_label, columns=['domain', 'label'])
df_domain.to_csv('domain.csv', index=False)