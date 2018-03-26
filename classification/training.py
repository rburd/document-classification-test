import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#load mortgage dataset
data = pd.read_csv("~/Documents/HeavyWater Problem/document_classification_test/dataset/shuffled-full-set-hashed.csv", header = 0)

#extract labels
labels = data['Label'].values

#extract raw data
rawData = data['Raw Data'].values


#divide labels and raw data into testing and training sets
train, test, train_labels, test_labels = train_test_split(rawData, labels, test_size=0.33, random_state=42)


countVector = CountVectorizer();
data_counts = countVector.fit_transform(train.astype('U'))

tf_transformer = TfidfTransformer(use_idf=False).fit(data_counts)
data_tf = tf_transformer.transform(data_counts)

#train Naive Bayes model
nb = LinearSVC()
model = nb.fit(data_tf, train_labels)

new_data_counts = countVector.transform(test.astype('U'))
tf_transformer = TfidfTransformer(use_idf=False).fit(new_data_counts)
new_data_tf = tf_transformer.transform(new_data_counts)

#predict
predictions = nb.predict(new_data_tf)
print(test_labels)
print(predictions)

# Evaluate accuracy
print(np.mean(predictions == test_labels))


