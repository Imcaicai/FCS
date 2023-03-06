import re
import threading

from lxml import etree
import requests
import pandas as pd
import datetime as dt
import numpy as np
import openpyxl as op

lock = threading.RLock()
service_url = "https://whois.chinaz.com/"
dataset_path = '../data/train_dataset_1.csv'
whois_feature_path = 'whois_feature_set.npy'
cn2en = {"更新时间": "updated_date", "创建时间": "creation_date", "过期时间": "expiration_date"}
bg = op.load_workbook(r"whois_feature_set.xlsx")
sheet = bg["Sheet1"]
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}

def get_whois_info(url):
    whois_info = {"updated_date": "nil", "creation_date": "nil", "expiration_date": "nil"}
    final_url = service_url + url
    doc = requests.get(final_url, headers=headers, timeout=20)
    doc.encoding = 'utf-8'
    doc = doc.text
    html = etree.HTML(doc)
    root_xpath = '//*[@id="whois_info"]'
    if len(html.xpath(root_xpath)):
        whois_info_len = len(html.xpath(root_xpath + "/li"))
        if whois_info_len:
            for i in range(whois_info_len):
                index = i + 1
                ele_xpath = root_xpath + "/li[{}]".format(index)
                title = html.xpath(ele_xpath + "/div[1]")[0].text
                if title in cn2en:
                    content = html.xpath(ele_xpath + "/div[2]/span")[0].text
                    content = content.replace(content[4], '-')
                    content = content.replace(content[7], '-')
                    content = content.replace(content[-1], '')
                    whois_info[cn2en[title]] = content
            if whois_info["updated_date"] == "nil":
                whois_info["updated_date"] = whois_info["creation_date"]
    print(url + ", " + whois_info.__str__())
    return whois_info

def process(url_list, begin, end):
    cnt = 1
    for url in url_list[begin: end]:
        data = [url]
        single_info_dict = get_whois_info(url)
        if "nil" in single_info_dict.values():
            for i in range(3):
                data.append(0)
        else:
            updated_date = dt.datetime.strptime(single_info_dict["updated_date"], "%Y-%m-%d").date()
            creation_date = dt.datetime.strptime(single_info_dict["creation_date"], "%Y-%m-%d").date()
            expiration_date = dt.datetime.strptime(single_info_dict["expiration_date"], "%Y-%m-%d").date()
            now_date = dt.date.today()
            data.append((now_date - updated_date).days)
            data.append((now_date - creation_date).days)
            data.append((expiration_date - creation_date).days)
        for i in range(1, len(data) + 1):
            sheet.cell(begin + cnt, i, data[i - 1])
        cnt += 1


class myThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, id, url_list, begin, end):
        threading.Thread.__init__(self)
        self.id = id
        self.url_list = url_list
        self.begin = begin
        self.end = end

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        process(self.url_list, self.begin, self.end)



if __name__ == '__main__':
    # 读取数据集
    df = pd.read_csv(dataset_path, header=0)
    # 获得url_list
    url_list = df['url']

    update2now = []
    create2now = []
    create2expire = []

    ts = []

    for i in range(len(url_list) // 100 + 1):
        begin = i * 100
        end = min(len(url_list), (i + 1) * 100)
        t = myThread(i, url_list, begin, end)
        ts.append(t)

    [t.start() for t in ts]
    [t.join() for t in ts]

    bg.save("whois_feature_set.xlsx")

