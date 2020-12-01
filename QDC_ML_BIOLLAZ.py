import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

df_data = pd.read_csv('C:/Users/guibi/PycharmProjects/QDC_group7/QDC_git/DATASET_prepared.csv', index_col=0, header=0,
                      low_memory=False)

df_data = df_data.drop(columns=["FL_CLI_USES_BANK_APP_M", "FL_CLI_USES_BANK_WEB_M", "QT_TRANSACTIONS_APP_3M",
                                "QT_TRANSACTIONS_WEB_3M", "FL_CLI_USES_BANK_APP_12M", "FL_CLI_USES_BANK_WEB_12M"])

df_sub_1 = df_data.filter(regex='^CD_GENDER|^QT_AGE|CD_SCOLARITY|CD_CLI_DISTRICT_ADDRESS|IS_TARGET', axis=1)


pd.set_option("display.max_columns", None)
#print(df_sub.head())

######
###### ML AND STUFF (PROTOTYPE)
######

X = df_sub_1.drop('IS_TARGET', axis=1)
y = df_sub_1.IS_TARGET

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = DecisionTreeClassifier(max_depth=5)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=X.columns, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('MILLENIUM.png')
Image(graph.create_png())
