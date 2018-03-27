import pandas as pd
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

#load mortgage dataset
data = pd.read_csv("~/Documents/Heavy Water/document_classification_test/dataset/shuffled-full-set-hashed.csv", header = 0)

#extract labels
labels = data['Label'].values

#extract raw data
rawData = data['Raw Data'].values


#divide labels and raw data into testing and training sets
train, test, train_labels, test_labels = train_test_split(rawData, labels, test_size=0.33, random_state=42)

pipeline = Pipeline([
    ('vector', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('nb', LinearSVC())
])

pipeline.fit(train.astype('U'), train_labels)

#predict
predictions = pipeline.predict(test.astype('U'))
print(test_labels)
print(predictions)

# Evaluate accuracy
print(np.mean(predictions == test_labels))

#Persist model on file
joblib.dump(pipeline, "model.p")


