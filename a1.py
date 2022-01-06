#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
np.random.seed(550)


# In[2]:


# load two data set, and label positive case as one and negative case as zero
data_positive = np.loadtxt("rt-polarity.pos", dtype='str', delimiter='\n', encoding='latin-1')
# print(data_positive)
ones = np.ones(data_positive.shape[0], int)
data_positive = np.c_[data_positive, ones]
data_negative = np.loadtxt("rt-polarity.neg", dtype='str', delimiter='\n', encoding='latin-1')
zeros = np.zeros(data_negative.shape[0], int)
data_negative = np.c_[data_negative, zeros]
# Concatenate two data frame
data = np.r_[data_positive, data_negative]
# Randomly shuffle the whole dataset
np.random.shuffle(data)
# Split to training and test set
training_set, test_set = train_test_split(data, test_size=0.15)


# In[3]:


# Define a general experiment procedure
def experiment(train_X, train_Y, test_X, test_Y, n_splits=5):
    # Use 5 cross validation by default
    kf = KFold(n_splits=n_splits)
    avg_mse_validation = 0
    avg_mse_train = 0
    avg_accuracy_validation = 0
    avg_accuracy_train = 0
    model = LogisticRegression(solver='liblinear')
    for train_indices, validation_indices in kf.split(training_set):
        train_features = train_X[train_indices]
        train_labels = np.array([int(l) for l in train_Y[train_indices]])
        validation_features = train_X[validation_indices]
        validation_labels = np.array([int(l) for l in train_Y[validation_indices]])
        model.fit(train_features, train_labels)
        validation_prediction = model.predict(validation_features)
        train_prediction = model.predict(train_features)
        mse_validation = mean_squared_error(validation_labels, validation_prediction)
        mse_train = mean_squared_error(train_labels, train_prediction)
        accuracy_validation = accuracy_score(validation_labels, validation_prediction)
        accuracy_train = accuracy_score(train_labels, train_prediction)
        avg_mse_validation += mse_validation
        avg_mse_train += mse_train
        avg_accuracy_validation += accuracy_validation
        avg_accuracy_train += accuracy_train
    model.fit(train_X, train_Y)
    test_prediction = model.predict(test_X)
    test_prediction = np.array([int(l) for l in test_prediction])
    test_Y = np.array([int(l) for l in test_Y])
    test_mse = mean_squared_error(test_Y, test_prediction)
    test_accuracy = accuracy_score(test_Y, test_prediction)
    print("avg_mse_validation", round(avg_mse_validation / 5, 6))
    print("avg_mse_train", round(avg_mse_train / 5, 6))
    print("avg_accuracy_validation", round(avg_accuracy_validation / 5, 6))
    print("avg_accuracy_train", round(avg_accuracy_train / 5, 6))
    print("test_mse", round(test_mse, 6))
    print("test_accuracy", round(test_accuracy, 6))


# In[4]:


#Use uni-gram as the first method
review = []
for row in range(data.shape[0]):
    review.append(data[row][0])
vectorizer = CountVectorizer()
vectorizer.fit(review)
training_review = []
for row in range(training_set.shape[0]):
    training_review.append(training_set[row][0])
test_review = []
for row in range(test_set.shape[0]):
    test_review.append(test_set[row][0])
train_X = vectorizer.transform(training_review)
train_Y = np.ravel(np.array(training_set.take([1], axis=1)))
test_X = vectorizer.transform(test_review)
test_Y = np.ravel(np.array(test_set.take([1], axis=1)))
print("Result of Unigram Experiment: ")
experiment(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


# In[5]:


# Use uni-gram combined PorterStemmer as the second method
porter = PorterStemmer()
def stemming(text):
    return ' '.join(porter.stem(word) for word in text.split())
review = []
for row in range(data.shape[0]):
    review.append(stemming(data[row][0]))
vectorizer = CountVectorizer()
vectorizer.fit(review)
training_review = []
for row in range(training_set.shape[0]):
    training_review.append(stemming(training_set[row][0]))
test_review = []
for row in range(test_set.shape[0]):
    test_review.append(stemming(test_set[row][0]))
train_X = vectorizer.transform(training_review)
train_Y = np.ravel(np.array(training_set.take([1], axis=1)))
test_X = vectorizer.transform(test_review)
test_Y = np.ravel(np.array(test_set.take([1], axis=1)))
print("Result of Unigram combined Stemming Experiment: ")
experiment(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


# In[6]:


# Use uni-gram combined Stop word removing as the third method
stop = stopwords.words('english')
def removing(text):
    return ' '.join(word for word in text.split() if word not in stop)
review = []
for row in range(data.shape[0]):
    review.append(removing(data[row][0]))
vectorizer = CountVectorizer()
vectorizer.fit(review)
training_review = []
for row in range(training_set.shape[0]):
    training_review.append(removing(training_set[row][0]))
test_review = []
for row in range(test_set.shape[0]):
    test_review.append(removing(test_set[row][0]))
train_X = vectorizer.transform(training_review)
train_Y = np.ravel(np.array(training_set.take([1], axis=1)))
test_X = vectorizer.transform(test_review)
test_Y = np.ravel(np.array(test_set.take([1], axis=1)))
print("Result of Unigram combined Stopwords removing Experiment: ")
experiment(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


# In[7]:


# Use bi-gram as the fourth method
review = []
for row in range(data.shape[0]):
    review.append(data[row][0])
vectorizer = CountVectorizer(ngram_range=(2,2))
vectorizer.fit(review)
training_review = []
for row in range(training_set.shape[0]):
    training_review.append(training_set[row][0])
test_review = []
for row in range(test_set.shape[0]):
    test_review.append(test_set[row][0])
train_X = vectorizer.transform(training_review)
train_Y = np.ravel(np.array(training_set.take([1], axis=1)))
test_X = vectorizer.transform(test_review)
test_Y = np.ravel(np.array(test_set.take([1], axis=1)))
print("Result of Bigram Experiment: ")
experiment(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


# In[8]:


# Trigram as the fiveth method
review = []
for row in range(data.shape[0]):
    review.append(data[row][0])
vectorizer = CountVectorizer(ngram_range=(3,3))
vectorizer.fit(review)
training_review = []
for row in range(training_set.shape[0]):
    training_review.append(training_set[row][0])
test_review = []
for row in range(test_set.shape[0]):
    test_review.append(test_set[row][0])
train_X = vectorizer.transform(training_review)
train_Y = np.ravel(np.array(training_set.take([1], axis=1)))
test_X = vectorizer.transform(test_review)
test_Y = np.ravel(np.array(test_set.take([1], axis=1)))
print("Result of Trigram Experiment: ")
experiment(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


# In[9]:


# Lemmatization as the sixth method
lemmatizer = WordNetLemmatizer()
def lemmatization(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())
review = []
for row in range(data.shape[0]):
    review.append(lemmatization(data[row][0]))
vectorizer = CountVectorizer()
vectorizer.fit(review)
training_review = []
for row in range(training_set.shape[0]):
    training_review.append(lemmatization(training_set[row][0]))
test_review = []
for row in range(test_set.shape[0]):
    test_review.append(lemmatization(test_set[row][0]))
train_X = vectorizer.transform(training_review)
train_Y = np.ravel(np.array(training_set.take([1], axis=1)))
test_X = vectorizer.transform(test_review)
test_Y = np.ravel(np.array(test_set.take([1], axis=1)))
print("Result of Unigram combined Lemmatization Experiment: ")
experiment(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

