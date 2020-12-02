import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

df_data = pd.read_csv('C:/Users/guibi/PycharmProjects/QDC_group7/QDC_git/DATASET_prepared_GENY.csv', index_col=0, header=0,
                      low_memory=False)

# df_data = df_data.drop(columns=["FL_CLI_USES_BANK_APP_M", "FL_CLI_USES_BANK_WEB_M", "QT_TRANSACTIONS_APP_3M",
#                                "QT_TRANSACTIONS_WEB_3M", "FL_CLI_USES_BANK_APP_12M", "FL_CLI_USES_BANK_WEB_12M"])

df_sub_1 = df_data.filter(regex='^CD_GENDER|^QT_AGE|CD_SCOLARITY|CD_CLI_DISTRICT_ADDRESS|IS_TARGET', axis=1)

df_data['QT_TRANSACTIONS_APP_3M_MORE5'] = np.where(df_data['QT_TRANSACTIONS_APP_3M'] >= 5, 1, 0)

df_sub_cleaned = df_data.drop(columns=["QT_OPER_BRANCH_3M", "QT_OPER_ATM_3M", "QT_OPER_DIGITAL_3M",
                                       "QT_OPER_OTHERS_3M", "PC_OPER_DIGITAL_3M", "PC_OPER_BRANCH_3M", "PC_OPER_ATM_3M",
                                       "FL_CLI_USES_BANK_APP_12M", 'FL_CLI_USES_BANK_APP_M',
                                       "FL_CLI_USES_BANK_WEB_M", "QT_TRANSACTIONS_APP_3M",
                                       "QT_TRANSACTIONS_WEB_3M", "IS_TARGET", "FL_CLI_USES_BANK_WEB_12M",
                                       "FL_DIGITAL_STATEMENT"])

df_sub_cleaned = df_sub_cleaned.drop(columns=df_sub_cleaned.filter(regex='^CD_CLI_CHANNEL_PREFERENCE', axis=1))

#df_sub_cleaned_under30 = df_sub_cleaned[df_sub_cleaned.QT_AGE <= 30]
pd.set_option("display.max_columns", None)
# print(df_sub.head())

######
###### ML AND STUFF (PROTOTYPE)
######
# print(df_sub_2.FL_CLI_USES_BANK_APP_M.value_counts())
X = df_sub_cleaned.drop('QT_TRANSACTIONS_APP_3M_MORE5', axis=1)
y = df_sub_cleaned.QT_TRANSACTIONS_APP_3M_MORE5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=500)
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
print("Accuracy Tree:", metrics.accuracy_score(y_test, y_pred_clf))

# KNN = KNeighborsClassifier(n_neighbors=3)
# KNN.fit(X_train, y_train)
# y_pred_KNN = KNN.predict(X_test)
# print("Accuracy KNN:", metrics.accuracy_score(y_test, y_pred_KNN))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=X.columns, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('QDC_ML_BIOLLAZ_5APP3MT_GENY.png')
Image(graph.create_png())
