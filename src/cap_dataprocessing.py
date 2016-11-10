import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from __future__ import division
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
from nltk.stem import SnowballStemmer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def process_file(filename):
    #separate the labels from the text
    #input:  fully qualified filename
    #return: Data = X and labels = y
    label = []
    text=[]
    with open(filename) as f:
        lines = f.readlines()
        for i in lines:
            label.append(i[0])
            text.append(i[1:])
    X = np.array(text)
    y = np.array(label)
    return X,y


def do_stemming(X):
    stemmed = []
    for item in X:
        stemmed.append(SnowballStemmer('english').stem(f))
    return stemmed

def train_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test

def vectorize(X):
    v = TfidfVectorizer(stop_words='english',strip_accents='unicode', lowercase=True)
    vector = v.fit_transform(X).toarray()
    #words = v.get_feature_names()
    return v

def classify(X,y):
    classifier = MultinomialNB()
    classifier.fit(train_vectors, y_train)
    return classifier

def clean_data(X):
    # Function: lowercase (also completed in vecorizer),
    #           remove special characters, remove numbers
    # Input: numpy array of strings of text
    # Output:  numpy string of cleansed text

    text = [re.sub(r'[^\w\s\d]','',h.lower()) for h in X]
    regex = re.compile('[^a-zA-Z]')
    l=[]
    for i in text:
        l.append(regex.sub(' ', i))
    return np.array(l)


# def pipeline(X,y):
#     pipeline = Pipeline([
#         ('vectorizer',  TfidfVectorizer()),
#         ('classifier',  MultinomialNB()) ])
#
#     pipeline.fit(X_train, y_train)
#     pipeline.predict(examples) # ['spam', 'ham']

def score(classifier,X,y):
    #Input: classifer
    #       data vectors
    #       labels
    #Output: score
    # cross_val_scores = cross_val_score(clf, train_vectors, y_train,  cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # nltk.classify.accuracy(cl,test_vectors)
    # v_probs = clf.predict_proba(test_vectors)[:, 1]
    score = classifier.score(X,y)
    return score


def predict(classifier, X):
    #input classifier  and text to predict
    #Return: prediction
    predictions= classifier.predict(d)
    return prediction

# examples = ['hate hate hate', "great fun awsome", "sucks"]

def plot_roc(probs, y_true, title, xlabel, ylabel):
    # ROC
    tpr, fpr, thresholds = roc_curve(v_probs, y_test)

    plt.hold(True)
    plt.plot(fpr, tpr)

    # 45 degree line
    xx = np.linspace(0, 1.0, 20)
    plt.plot(xx, xx, color='red')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()


def plot_datatwin(p,n,tme):
#
    x = np.linspace(0,len(neg), tme)
    # plt.plot(x, neg,x, pos)
    # plt.plot(x, reg_pos, x, reg_neg)
    total = []

    for i in range(0,len(pos)):
        total.append((pos[i] + neg[i]))

    fig, ax1 = plt.subplots()
    s1 = pos
    ax1.plot(x, pos, 'b-')
    ax1.plot(x, neg, 'r-')
    ax1.plot(x, total, 'y-')
    ax1.set_xlabel('Unit of Time')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Number of Tweets', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')


    plt.show()

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''

    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []
    labels = np.array(map(float, labels))
    print y_float
    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()


def plot_importance(clf, X, max_features=10):
    '''Plot feature importance'''
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # Show only top features
    pos = pos[-max_features:]
    feature_importance = (feature_importance[sorted_idx])[-max_features:]
    feature_names = (X.columns[sorted_idx])[-max_features:]

    plt.barh(pos, feature_importance, align='center')
    plt.yticks(pos, feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')


#plot_roc(v_probs, y_test,"ROC plot of churn data","False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity, Recall)")
plot_datatwin(pos,neg,len(neg))
