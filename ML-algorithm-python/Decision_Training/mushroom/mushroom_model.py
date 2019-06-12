import graphviz
import pandas as pd

from sklearn import preprocessing
from sklearn import tree


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


def replace_data(df, col_name=None):

    def replace(data):
        result = []
        for d in data:
            if d == '?':
                result.append(['n'])
                continue
            elif d == 0:
                result.append(['e'])
                continue
            elif d == 1:
                result.append(['p'])
                continue
            result.append([d])
        return result

    copied_df = df.copy()
    old_list = list(copied_df[col_name])
    new_list = replace(old_list)
    copied_df[[col_name]] = new_list
    return copied_df


def reindex(df, targets):
    for index, name in enumerate(targets):
        df.rename(index={index: name})


'''
    A data structure of train.tsv is like
    id,Y,cap-shape,...
    0,p,f,....
'''
df = pd.read_csv('./train.tsv', delimiter='\t')
test_df = pd.read_csv('./test.tsv', delimiter='\t')
test_data_index = test_df['id'].values

# new_df = replace_data(df, 'stalk-root')
# new_test_df = replace_data(test_df, 'stalk-root')

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
test_data_le = pd.get_dummies(test_df, drop_first=True,
                              columns=dummy_titles)


transform_data(category_titles, mr_data_le)
# mr_data_le.to_csv(path_or_buf='transformed.csv')
transform_data(test_category_titles, test_data_le)
test_feature_names = list(test_data_le.columns)
remove_col(['id'], test_feature_names)
# print(test_feature_names)

# get Y
y_data = mr_data_le['Y'].values

# get title of X
title_names = list(mr_data_le.columns)
remove_col(['Y', 'id'], title_names)

test_title_names = list(test_data_le.columns)
remove_col(['id'], test_title_names)

# get X
x_data = mr_data_le[title_names].values
test_x_data = test_data_le[test_title_names].values

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_data, y_data)

predict_result = pd.DataFrame(
    clf.predict(test_x_data), columns=['result'], index=test_data_index)

# print chart flow
graph = graphviz.Source(
    tree.export_graphviz(clf, out_file=None, feature_names=test_feature_names))
graph.format = 'png'
graph.render('dtree_render', view=True)

transformed = replace_data(predict_result, 'result')
transformed.reindex(index=test_data_index)
transformed.to_csv(path_or_buf='predict_result.csv', header=False)
