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

# ����
print(d_train.shape)
print(d_test.shape)

# �f�[�^�v��
print(d_train.head())
print(d_train.describe())
print(d_train.dtypes)

# �����l
print(d_train.info())
d_train.isnull().sum()
# mssno.bar(d_train,color='g',figsize=(16,5),fontsize=12)
sns.heatmap(d_train.isnull(), cbar=False)


# Categoral analysys - describe()�ŕ\������Ȃ������J�����Ώ�
## �񍀒l martial, default, housing, loan
print(d_train['default'].value_counts())
print(d_train['housing'].value_counts())
print(d_train['loan'].value_counts())

## Less than 10�J�e�S�� 
print(d_train['marital'].value_counts())
print(d_train['contact'].value_counts())
print(d_train['education'].value_counts())

## more than 10�J�e�S��
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


# ����
d_train.corr()
sns.heatmap(d_train.corr())

# 3�V�O�}�@
#   - ���K���z���݂���
#   - low  = mean[col] - 3 * sigma[col]
#   - high = mean[col] + 3 * sigma[col]


# �O��l�̑Ώ����@
#  - �O��l����菜��
#  - ���ϒl���邢�͒����l�Ŗ��߂�
#  - �听������

# �C�ɂȂ�l
## �N���
bining = pd.cut(d_train['age'],list(range(10,100,10)))
sns.countplot(x=bining, hue='y',data=d_train)

## housing��
sns.countplot(x='contact',hue='y',data=d_train)


## �N��ʁE�E��ʂɑ΂�������J�ݎ҂̊��������߂�
#10�Ζ��Ƀr�j���O
bining = pd.cut(d_train['age'],list(range(10,100,10)))
age_job = pd.crosstab(bining,d_train['job'], margins=True)

#�����J�ݎ҂𒊏o(y=1)
open_account = d_train[d_train['y']==1]
age_job_open = pd.crosstab(bining, open_account['job'],margins=True)

graph = age_job_open/age_job
graph.index = ['10��','20��','30��','40��','50��','60��','70��','80��','All']
graph


# age��job�ŃN���X�^�u
age_bining = pd.cut(d_train['age'],list(range(10,100,10))) #trainX�͋��t�f�[�^�̐����ϐ�
age_job = pd.crosstab(age_bining,d_train['job'],margins=True)
open_account = d_train[d_train['y']==1]
age_job = pd.crosstab(age_bining,age_job['job'],margins=True)
graph = age_job_open/age_job
graph.index = ['10��','20��','30��','40��','50��','60��','70��','80��','All']
graph
