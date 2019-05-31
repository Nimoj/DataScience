import pandas
from sklearn import tree

'''
    A data structure of train.tsv is like
    id,Y,cap-shape,...
    0,p,f,....
'''
df = pandas.read_csv('./train.tsv', delimiter='\t')

# get Y
y_data = df['Y'].values

# get title of X
title_names = list(df.columns)
title_names.remove('id')
title_names.remove('Y')
# get X
x_data = df[title_names].values

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_data, y_data)
