import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load data
breast_cancer_df = pd.read_csv('./breast_cancer_wisconsin.csv')

# Clean data
breast_cancer_df.drop('Unnamed: 32', axis=1, inplace=True)
breast_cancer_df.diagnosis.value_counts()

# Objective variable
y = breast_cancer_df.diagnosis.apply(lambda d: 1 if d == 'M' else 0)

# Explanatory variable
X = breast_cancer_df.ix[:, 'radius_mean':]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression
logistic = LogisticRegressionCV(cv=10, random_state=0)
logistic.fit(X_train_scaled, y_train)

# Testing
print('Train score: {:.3f}'.format(logistic.score(X_train_scaled, y_train)))
print('Test score: {:.3f}'.format(logistic.score(X_test_scaled, y_test)))
print('Confustion matrix:\n{}'.format(
    confusion_matrix(y_true=y_test, y_pred=logistic.predict(X_test_scaled))))

# Dimension reduction
plt.figure(figsize=(20, 20))
seaborn.heatmap(pd.DataFrame(X_train_scaled).corr(), annot=True)

# Principal component analysis
pca = PCA(n_components=30)
pca.fit(X_train_scaled)
plt.bar(
    [n for n in range(1, len(pca.explained_variance_ratio_)+1)],
    pca.explained_variance_ratio_)

# Try to reduce dimension to two
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
print('X_train_pca shape: {}'.format(X_train_pca.shape))

# Contribution rate
print('explained variance ratio: {}'.format(pca.explained_variance_ratio_))

# Scatter plot
temp = pd.DataFrame(X_train_pca)
temp['Result'] = y_train.values
# benign
b = temp[temp['Result'] == 0]
# malignance
m = temp[temp['Result'] == 1]
plt.scatter(x=b[0], y=b[1], marker='circle')
plt.scatter(x=m[0], y=m[1], marker='square')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

# TODO:plot decison boundary
