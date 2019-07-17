# Accuracy on training set: 0.914
# Accuracy on test set: 0.902

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
# import missingno as mssno

# data loading
d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')
d_test_index = d_test["id"]

# profiling
import pandas_profiling
pandas_profiling.ProfileReport(d_train)

# 件数
print(d_train.shape)
print(d_test.shape)

# データ要約
print(d_train.head())
print(d_train.describe())
print(d_train.dtypes)

# 欠損値
print(d_train.info())
d_train.isnull().sum()
# mssno.bar(d_train,color='g',figsize=(16,5),fontsize=12)
sns.heatmap(d_train.isnull(), cbar=False)


# Categoral analysys - describe()で表示されなかったカラム対象
## 二項値 martial, default, housing, loan
print(d_train['default'].value_counts())
print(d_train['housing'].value_counts())
print(d_train['loan'].value_counts())

## Less than 10カテゴリ 
print(d_train['marital'].value_counts())
print(d_train['contact'].value_counts())
print(d_train['education'].value_counts())

## more than 10カテゴリ
print(d_train['job'].value_counts())

# Data Visualization - categoral
cat = ['default', 'housing', 'loan', 'marital', 'contact', 'education', 'job']
fig = plt.figure(figsize=(30, 10))
for i in range(0, len(cat)):
  fig.add_subplot(3,3,i+1)
  sns.countplot(x=cat[i], data=d_train); 

# Data Visualization - 
cat = ['age', 'balance']
fig = plt.figure(figsize=(30, 10))
for i in range(0, len(cat)):
  fig.add_subplot(3,3,i+1)
  sns.distplot(d_train[cat[i]], kde=False)


# Data Visualization - Categoral - Layer
def simple_cat_v(label_x):
  plttitle = "Y/Non-Y Distribution according to " + label_x
  plt.subplots(figsize = (15,8))
  ax = sns.barplot(x = label_x, y= "y",data=d_train, palette = "dark",linewidth=2 )
  plt.title(plttitle, fontsize = 25)
  plt.ylabel("% of Y", fontsize = 15)
  plt.xlabel(label_x,fontsize = 15)

def cross_cat_v(label_x, cat):
  for i in cat:
    if label_x == i:
      continue
    plttitle = "Y/Non-Y Distribution according to " + label_x + " by" + i
    plt.subplots(figsize = (15,8))
    ax = sns.barplot(x = label_x, y= "y", hue = i, data=d_train, palette = "dark",linewidth=2 )
    plt.title(plttitle, fontsize = 25)
    plt.ylabel("% of Y", fontsize = 15)
    plt.xlabel(label_x,fontsize = 15)


cat = ['default', 'housing', 'loan', 'marital', 'contact']
for i in cat:
  simple_cat_v(i)


for i in cat:
  cross_cat_v(i, cat)


# 相関
d_train.corr()
sns.heatmap(d_train.corr())

# 3シグマ法
#   - 正規分布をみつける
#   - low  = mean[col] - 3 * sigma[col]
#   - high = mean[col] + 3 * sigma[col]


# 外れ値の対処方法
#  - 外れ値を取り除く
#  - 平均値あるいは中央値で埋める
#  - 主成分分析

# 気になる値
## 年齢別
bining = pd.cut(d_train['age'],list(range(10,100,10)))
sns.countplot(x=bining, hue='y',data=d_train)

## housing別
sns.countplot(x='contact',hue='y',data=d_train)


## 年齢別・職種別に対する口座開設者の割合を求める
#10歳毎にビニング
bining = pd.cut(d_train['age'],list(range(10,100,10)))
age_job = pd.crosstab(bining,d_train['job'], margins=True)

#口座開設者を抽出(y=1)
open_account = d_train[d_train['y']==1]
age_job_open = pd.crosstab(bining, open_account['job'],margins=True)

graph = age_job_open/age_job
graph.index = ['10代','20代','30代','40代','50代','60代','70代','80代','All']
graph


# ageとjobでクロスタブ
age_bining = pd.cut(d_train['age'],list(range(10,100,10))) #trainXは教師データの説明変数
age_job = pd.crosstab(age_bining,d_train['job'],margins=True)
open_account = d_train[d_train['y']==1]
age_job = pd.crosstab(age_bining,age_job['job'],margins=True)
graph = age_job_open/age_job
graph.index = ['10代','20代','30代','40代','50代','60代','70代','80代','All']
graph
