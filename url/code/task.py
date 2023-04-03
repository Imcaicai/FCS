import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin

result=[['111111',0],['222222',0]]

# �벻���� url �ϲ�
abnormal = np.array(pd.read_csv('abnormal.csv'))
# au = abnormal['url']
# al = abnormal['label']
url = []
label = []
# for i in au:
#     url.append(i)
# for i in al:
#     label.append(i)
for i in abnormal:
    url.append(i[0])
    label.append(i[1])
for i in result:
    url.append(i[0])
    label.append(i[1])

# ����ȡ�����ѵ����
sample = np.array((url,label)).T
sample_set = pd.DataFrame(sample, columns=['url','label'])
sample_set.to_csv('train.csv', index=False)