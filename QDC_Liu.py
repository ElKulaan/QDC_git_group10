# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:17:06 2020

@author: Chiahui Liu
"""
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble 
from sklearn.model_selection import GridSearchCV
from typing import Union
from sklearn import metrics
from sklearn.metrics import auc,roc_curve
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassfier
from matplotlib import pyplot
#data loading
df_full_target=pd.read_csv('DATASET_prepared_1.csv')
print(df_full_target.head(3))

#visualizing missing value
sb.heatmap(df_full_target.isnull())

#seperate the datafram into x and y
X = df_full_target.drop('IS_TARGET', axis=1)
y = df_full_target.IS_TARGET

# random forest model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
rf_clf = ensemble.RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf_clf = rf_clf.predict(X_test)
print("Accuracy Tree:", metrics.accuracy_score(y_test, y_pred_rf_clf))

#plot roc
fpr, tpr, _ = roc_curve(y_test, y_pred_rf_clf, pos_label=rf_clf.classes_[1])
roc_display_1 = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

#plot recall score
prec, recall, _ = precision_recall_curve(y_test, y_pred_rf_clf,pos_label=rf_clf.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

# feature improtance 
feat_labels = df_full_target.columns[1:]
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature Importance")

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


    # Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=feat_labels[indices[f]], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
graph.write_png('1.png')
Image(graph.create_png())


#############
#param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[5, 6, 7, 8],    
    'n_estimators':[11,13,15],  
    'max_features':[0.3,0.4,0.5],
    'min_samples_split':[4,8,12,16]  
}

#rfc = ensemble.RandomForestClassifier()
#rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=5)
#rfc_cv.fit(X_train, y_train)
#test_est = rfc_cv.predict(X_test)

# test on decision tree model
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=500)
#clf.fit(X_train, y_train)
#y_pred_clf = clf.predict(X_test)
#print("Accuracy Tree:", metrics.accuracy_score(y_test, y_pred_clf))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=500)
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
print("Accuracy Tree:", metrics.accuracy_score(y_test, y_pred_clf))


print(feat_labels)
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))