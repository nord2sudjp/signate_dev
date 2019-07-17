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


# decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=7, random_state=0)
clf = clf.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))

# export tree as dot
with open('graph.dot', 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

# Predictions
predictions = clf.predict(d_test)
signate_submission = pd.DataFrame({"id":d_test_index, "y":predictions})
signate_submission.to_csv("signate_bank01_dc_01.csv", index = False)

# Feature Importance
print("Feature importances:")
print(tree.feature_importances_)

n_features = X_train.shape[1]
np.arange(n_features)
plt.barh(np.arange(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), d_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)

# decsion tree - hyper parameter adjustment - max_depth

d_train1 = d_train.sample(5000)
X_train = d_train1.drop('y', axis=1)
y_train = d_train1.y

features = np.array(X_train)
targets = np.array(y_train)

MAX_DEPTH = 20
depths = range(1, MAX_DEPTH)
accuracy_scores = []

for depth in depths:
  predicted_labels = []
  # LOO 法で汎化性能を調べる
  loo = LeaveOneOut()
  for train, test in loo.split(features):
      # 学習に使うデータ
      train_data = features[train]
      target_data = targets[train]

      # モデルを学習させる
      clf = DecisionTreeClassifier()
      clf.fit(train_data, target_data)

      # テストに使うデータを正しく判定できるか
      predicted_label = clf.predict(features[test])
      predicted_labels.append(predicted_label)
  score = accuracy_score(targets, predicted_labels)
  print('max depth={0}: {1}'.format(depth, score))
  accuracy_scores.append(score)

X = list(depths)
plt.plot(X, accuracy_scores)

plt.xlabel('max depth')
plt.ylabel('accuracy rate')
plt.show()
