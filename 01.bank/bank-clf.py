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

# 前処理-列のドロップ
d_train = d_train.drop(['id', 'day'],axis=1)
d_test = d_test.drop(['id', 'day'], axis=1)

# 欠損値の修復
## カテゴリ - 1) 真ん中のクラス, 2) 他の値からの推測 3) 回帰分析
##     2) ax = sns.boxplot("Embarked","Fare", palette='rainbow', hue='Pclass',data=test)
## 連続値 - 1) 平均, 2) 層化した平均, 3) ランダム値, 4) 回帰分析

# 前処理-カテゴリデータの変換
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

labels = ['job','marital','education', 'default', 'housing', 'loan', 'contact','poutcome', 'month']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])
d_train.head()
d_test.head()

#前処理-新しい特徴量

# split
from sklearn.model_selection import train_test_split
X_train = d_train.drop('y', axis=1)
y_train = d_train.y
(X_train, X_test ,y_train, y_test) = train_test_split(X_train, y_train,  test_size = 0.3, random_state = 666)



#
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


dtree = DecisionTreeClassifier(max_depth=8)
dtree.fit(X_train,y_train)
print("Accuracy on training set - DTree: {:.3f}".format(dtree.score(X_train, y_train)))
print("Accuracy on test set - DTree: {:.3f}".format(dtree.score(X_test, y_test)))
dtree.score(X_train, y_train)
Y_pred = dtree.predict(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Accuracy on training set - LR: {:.3f}".format(logreg.score(X_train, y_train)))
print("Accuracy on test set - LR: {:.3f}".format(logreg.score(X_test, y_test)))
logreg.score(X_train, y_train)
Y_pred = logreg.predict(X_test)

svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on training set - SVC: {:.3f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set - SVC: {:.3f}".format(svc.score(X_test, y_test)))
svc.score(X_train, y_train)
Y_pred = svc.predict(X_test)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
print("Accuracy on training set - RF: {:.3f}".format(random_forest.score(X_train, y_train)))
print("Accuracy on test set - RF: {:.3f}".format(random_forest.score(X_test, y_test)))
random_forest.score(X_train, y_train)
Y_pred = random_forest.predict(X_test)

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
print("Accuracy on training set - GNB: {:.3f}".format(gaussian.score(X_train, y_train)))
print("Accuracy on test set - GNB: {:.3f}".format(gaussian.score(X_test, y_test)))
gaussian.score(X_train, y_train)
Y_pred = gaussian.predict(X_test)
