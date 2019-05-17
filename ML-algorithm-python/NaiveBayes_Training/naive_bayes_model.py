import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


spam_df = pd.read_csv('spam.csv', encoding='latin-1')
# Clean data
spam_df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

spam_df['v1'].value_counts()

X = pd.DataFrame(spam_df['v2'])
y = spam_df['v1'].apply(lambda s: 1 if s == 'spam' else 0)

# Train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)

# CoutVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X_train['v2'])

print('Vocabulary size: {}'.format(len(vectorizer.vocabulary_)))
print('Vocabulary content: {}'.format(vectorizer.vocabulary_))

# Feature vectorization
X_train_bow = vectorizer.transform(X_train['v2'])
X_test_bow = vectorizer.transform(X_test['v2'])

print('X_train_bow:\n{}'.format(repr(X_train_bow)))
print('X_test_bow:\n{}'.format(repr(X_test_bow)))

# Naive Bayes classifier
model = BernoulliNB()
model.fit(X_train_bow, y_train)

print('Train accuracy: {:.3f}'.format(model.score(X_train_bow, y_train)))
print('Test accuracy: {:.3f}'.format(model.score(X_test_bow, y_test)))