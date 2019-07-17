# Accuracy on training set: 0.914
# Accuracy on test set: 0.902

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

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




# decision tree model
from sklearn.ensemble import RandomForestClassifier

# cross validate
random_forest = RandomForestClassifier(n_estimators=100)
cv=cross_validate(random_forest, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=1)
print(cv)

random_forest = RandomForestClassifier(n_estimators=10)
cv=cross_validate(random_forest, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=1)
print(cv)

random_forest = RandomForestClassifier(n_estimators=5)
cv=cross_validate(random_forest, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=1)
print(cv)

# GridSearch
random_forest = RandomForestClassifier()
parameters = {"max_depth":[5,10,20], "n_estimators":[5,10,100,200], 'max_features':[1,None,'auto'],'min_samples_leaf':[3,4,5,6]}
gcv = GridSearchCV(random_forest, parameters, cv=5, scoring="roc_auc", n_jobs=1)
gcv.fit(X_train, y_train)
print(gcv.cv_results_)
train_score = gcv.cv_results_["mean_train_score"]
test_score = gcv.cv_results_["mean_test_score"]

print("Train Score", train_score)
print("Test Score", test_score)

plt.plot(train_score)
plt.plot(test_score)
plt.xticks([0,1,2,3,4],[1,3,5,10,100])

print(gcv.best_params_)

# fitting
predictions = gcv.predict_proba(d_test)
signate_submission = pd.DataFrame({"id":d_test_index, "y":predictions})
signate_submission.to_csv("signate_bank01_rf_01.csv", index = False)


# fitting
random_forest = RandomForestClassifier(n_estimators=100, max_depth=10)
random_forest.fit(X_train, y_train)
print("Accuracy on training set - RF: {:.3f}".format(random_forest.score(X_train, y_train)))
print("Accuracy on test set - RF: {:.3f}".format(random_forest.score(X_test, y_test)))
random_forest.score(X_train, y_train)
predictions = random_forest.predict(X_test)


# Predictions
predictions = random_forest.predict(d_test)
signate_submission = pd.DataFrame({"id":d_test_index, "y":predictions})
signate_submission.to_csv("signate_bank01_rf_01.csv", index = False)
