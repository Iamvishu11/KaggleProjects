# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# importing the dataset
train_df = pd.read_csv('labeledTrainData.tsv',header=0, delimiter = "\t", quoting = 3)
test_df = pd.read_csv('testData.tsv',header=0, delimiter = "\t", quoting = 3)

#Cleaning text of trainig set
from bs4 import BeautifulSoup  #Removing HTML Markup: The BeautifulSoup Package
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,25000):
    review = BeautifulSoup(train_df['review'][i])
    review = re.sub('[^a-zA-Z]', ' ', train_df['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X_train = cv.fit_transform(corpus).toarray()

# Fitting RandomForest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100)
classifier = classifier.fit(X_train, X_train["sentiment"] )

#Cleaning text of test set
corpus2 = []

for i in range(0,25000):
    review = BeautifulSoup(test_df['review'][i])
    review = re.sub('[^a-zA-Z]', ' ', test_df['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus2.append(review)

# Get a bag of words for the test set, and convert to a numpy array
X_test = cv.transform(corpus2).toarray()

# Fitting RandomForest to the Test set
result = classifier.predict(X_test)

# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
output = pd.DataFrame( data={"id":test_df["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )