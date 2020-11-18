import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus



pd.set_option("display.max_columns", None)

## OPEN BOTH DATASETS
df_full_data = pd.read_csv('D:/QDC/QTEM_DATABASE_csv.csv', index_col=0, header=0, low_memory=False)
df_full_target = pd.read_csv('D:/QDC/QTEM_TARGET_csv.csv', index_col=0, header=0, low_memory=False)

## PRINT BOTH DATASETS
# print("DATA")
# print(df_full_data)

## ADD A COLUMN (BOOLEAN encoded as an INTEGER) TO THE MAIN DATASET, CRITERIA --> IS THIS CLIENT A TARGET ?
df_full_data['IS_TARGET'] = np.where(df_full_data.index.isin(df_full_target.index), 1, 0)


def get_frequencies(data: pd.DataFrame, n_categories: int = None,
                    bins: int = None, dropna: bool = False
                    ):
    for (name, val) in data.iteritems():
        print('')
        print(name)
        vc = val.value_counts(ascending=False,
                              bins=bins,
                              dropna=dropna
                              )
        if n_categories is not None:
            if not isinstance(n_categories, int) or n_categories <= 0:
                raise TypeError(
                    'n_categories should be a strictly positive integer')
            if n_categories < len(vc):
                freq_others = vc.iloc[n_categories - 1:].sum()
                vc = vc.iloc[:n_categories - 1] \
                    .append(pd.Series({'others': freq_others}))
        print(pd.DataFrame({'absolute': vc,
                            'relative': vc / len(val) * 100,
                            },
                           index=vc.index
                           ).T)


##DESCRIPTION OF THE DATASET
def summary_data(data: pd.DataFrame):
    # FLAG
    print("\n FLAG")
    print(data.filter(regex=r"^FL_", axis=1).astype(bool).describe().transpose())

    # CODE
    print("\n CODE")
    print(data.filter(regex=r"^CD_", axis=1).astype(str).describe().transpose())

    # VALUE
    print("\n VALUE")
    print(data.filter(regex=r"^VL_", axis=1).astype(str).describe().transpose())

    # QUANTITY
    print("\n QUANTITY")
    print(data.filter(regex=r"^QT_", axis=1).astype(str).describe().transpose())

    # PERCENTAGE
    print("\n PERCENTAGE")
    print(data.filter(regex=r"^PC_", axis=1).describe().transpose())

    # NUMBER
    print("\n NUMBER")
    print(data.filter(regex=r"^NR_", axis=1).describe().transpose())

    print("\n NAN summary")
    print(summarize_na(data))


# percentage of missing data by variables
def summarize_na(df: pd.DataFrame) -> pd.DataFrame:
    nan_count = df.isna().sum()
    nan_pct = nan_count / len(df) * 100
    return pd.DataFrame({'nan_count': nan_count,
                         'nan_pct': nan_pct
                         }
                        )[nan_pct > 40]


# print(summarize_na(df_full_data))
summary_data(df_full_target)

# remove the data with more than "NAN" than the threshold defined in function "summarize_na"
df_full_data = df_full_data.drop(summarize_na(df_full_data).index, axis=1)

# print(df_full_data)

# split the data into numerical and categorical variables
df_cat = df_full_data.select_dtypes(include="object").copy()
df_num = df_full_data.select_dtypes(exclude="object").copy()


###
### IDEAS OF GRAPH
###
# PIE
tmp1 = df_full_data['IS_TARGET'].value_counts() / len(df_full_data) * 100
ax1 = tmp1.plot(kind='pie',
                title='Distribution of TARGET variable',
                autopct='{:02.2f}%'.format,
                legend=True,
                labeldistance=None,
                startangle=170,
                labels=['NOT TARGET', 'TARGET'],
                figsize=(5, 5)
                )
plt.show()


# HISTOGRAM
tmp2 = df_full_data['QT_AGE']
ax2 = tmp2.plot(kind='hist',
                title='Distribution of age of customer',
                bins=10,
                density=True,
                rwidth=.9,
                )
ax2.set(xlabel='Age (in years)')
plt.show()

# HORIZONTAL BAR
tmp3 = df_full_data['CD_CIVIL_STATUS'].value_counts() / len(df_full_data) * 100
ax = tmp3.plot(kind='barh',
               title="Distribution of customers civil status",
               width=.6,
               )
ax.grid(True, axis='x', color='lightgrey', linestyle='--')
ax.set(xlabel='Frequency (in %)')
plt.show()


#######
####### ML AND STUFF (PROTOTYPE)
#######

# def encode_onehot(series: pd.Series, drop_last: bool = True
#                   ) -> Union[pd.Series, pd.DataFrame]:
#     values = series.unique()
#     if drop_last:
#         values = values[:-1]
#     return pd.concat(((series == val).rename(val)
#                       for val in values
#                       ),
#                      axis=1
#                      ).squeeze()
#
#
# print("\n ENCODING GENDER")
# df_num['IS_FEMALE'] = encode_onehot(df_full_data['CD_GENDER'])
#
# print("\n NEW DATA")
# print(df_num)
# pd.set_option("display.max_rows", None)
# print(df_num.dtypes)
#
# df_num = df_num.dropna()
#
# X = df_num.drop('IS_TARGET', axis=1)
# y = df_num.IS_TARGET
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
# clf = DecisionTreeClassifier(max_depth=5)
#
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
#
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names=X.columns, class_names=['0', '1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('MILLENIUM.png')
# Image(graph.create_png())


