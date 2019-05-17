import graphviz

from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

names = ['sepal length', 'sepal width', 'petal length', 'petal width']

graph = graphviz.Source(
    tree.export_graphviz(clf, out_file=None, feature_names=names))
graph.format = 'png'
graph.render('dtree_render', view=True)
