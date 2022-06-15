# Adapted from https://www.kaggle.com/code/leandrodoze/sentiment-analysis-in-portuguese/notebook

import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import svm 


# Dataset
dataset = pd.read_csv('ignore_tweets.csv',encoding='utf-8')

# Instances and Classes
tweets = dataset["Text"].values

classes = dataset["Classificacao"].values

# Bag of Words
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('portuguese')
vectorizer = CountVectorizer(analyzer = "word", stop_words=stop_words)
freq_tweets = vectorizer.fit_transform(tweets)


# Using TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
freq_tweets = tfidf_transformer.fit_transform(freq_tweets)
freq_tweets.shape


modelo = svm.SVC(gamma=0.1, C=100)

# modelo = MultinomialNB()

modelo.fit(freq_tweets, classes)


# Validation
resultados = cross_val_predict(modelo, freq_tweets, classes, cv = 10)
print(resultados)

# Accuracy
print(metrics.accuracy_score(classes, resultados))

# Metrics
sentimentos = ["Positivo", "Negativo", "Neutro"]
print(metrics.classification_report(classes, resultados, labels=sentimentos))

# Confusion Matrix
print(pd.crosstab(classes, resultados, rownames = ["Real"], colnames=["Predito"], margins=True))


