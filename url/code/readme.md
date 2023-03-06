### 1 【13个】一起多分类

【数据】

- train_dataset_1：abnormal.csv所有数据  

【结果】

- Basic Feature Classification Results

  LR cross value:  0.6759251906423699

  Logistic Regression prediction:  0.6720822281167109

- Lexica Feature Set Classification Results

  LR cross value:  0.6803209345037611

  Logistic Regression prediction:  0.6840185676392573



### 2 【0和非0】二分类

【数据】

- train_dataset_2：abnormal.csv中非0标签都改为1

【结果】

- Basic Feature Classification Results 

  LR cross value:  0.9819266700067834

  Logistic Regression prediction:  0.9870689655172413

- Lexica Feature Set Classification Results 

  LR cross value:  0.9826729386634998 

  Logistic Regression prediction:  0.9860742705570292



### 3 【2和非2】二分类

【数据】

- train_dataset_3：abnormal.csv中去掉0标签，非2标签都改为13

【结果】

- Basic Feature Classification Results 

  LR cross value:  0.6026430960926946 

  Logistic Regression prediction:  0.6061525129982669

- Lexica Feature Set Classification Results 

  LR cross value:  0.6730570671540802 

  Logistic Regression prediction:  0.6798093587521664



### 4 【6、4、10、11、8、9和其他】分类

【数据】

- train_dataset_4：abnormal.csv中去掉0、2标签，非4标签都改为13

【结果】

- Basic Feature Classification Results 

  LR cross value:  0.3037446314339224 

  Logistic Regression prediction:  0.327991452991453  

- Lexica Feature Set Classification Results 

  LR cross value:  0.3799312879887565 

  Logistic Regression prediction:  0.42094017094017094



### 其他想法

- 合并文本&纯url的预测结果
  - 分别给出数据的预测概率，加权平均，选取概率最大的
  - 把文本特征转化为一些二进制特征，和纯url的特征放一起分类