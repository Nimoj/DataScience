import pandas as pd
from sklearn import tree
from sklearn import preprocessing


def transform_data(columns, df):
    for column in columns:
        selected_column = df[column]
        le = preprocessing.LabelEncoder()
        le.fit(selected_column)
        column_le = le.transform(selected_column)
        df[column] = pd.Series(column_le).astype('category')


def remove_col(rm_targets, data):
    for name in rm_targets:
        data.remove(name)


'''
    A data structure of train.tsv is like
    id,Y,cap-shape,...
    0,p,f,....
'''
df = pd.read_csv('./train.tsv', delimiter='\t')
test_df = pd.read_csv('./test.tsv', delimiter='\t')
dummy_titles = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                'gill-attachment', 'gill-color', 'stalk-shape', 'stalk-root',
                'stalk-surface-above-ring', 'stalk-surface-below-ring',
                'stalk-color-above-ring', 'stalk-color-below-ring',
                'veil-type', 'veil-color', 'ring-type', 'spore-print-color',
                'habitat']
category_titles = ['Y', 'gill-spacing', 'gill-size', 'ring-number',
                   'population']
test_category_titles = ['gill-spacing', 'gill-size',
                        'ring-number', 'population']

mr_data_le = pd.get_dummies(df, drop_first=True, columns=dummy_titles)
test_data_le = pd.get_dummies(test_df, drop_first=True, columns=dummy_titles)

transform_data(category_titles, mr_data_le)
transform_data(test_category_titles, test_data_le)

# get Y
y_data = mr_data_le['Y'].values

# get title of X
title_names = list(mr_data_le.columns)
remove_col(['Y', 'id'], title_names)

test_title_names = list(test_data_le.columns)
test_title_names.remove('id')
remove_col(['id'], test_title_names)

# get X
x_data = mr_data_le[title_names].values
test_x_data = test_data_le[test_title_names].values

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_data, y_data)
# print(list(clf.predict(test_x_data))
