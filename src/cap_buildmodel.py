from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
from nltk.stem import SnowballStemmer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import cPickle as pickle

class Cap_buildmodel(object):
    def __init__(self):
        #self.df = clean.load_and_clean('data.json')
        #self.text_features = ct.html_table(df['name'], df['description'])
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.json_model = None

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

def do_stemming(X):
    stemmed = []
    for item in X:
        stemmed.append(SnowballStemmer('english').stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = do_stemming(tokens)
    return stems

def train_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test

def vectorize_fit(X):
    v = TfidfVectorizer(tokenizer=tokenize,stop_words='english',strip_accents='unicode', lowercase=True)
    vectorizer = v.fit_transform(X) ## removed nparray()
    words = v.get_feature_names()
    #top_features(v,words,10)
    return v,vectorizer
# testing older code
def make_vecs(X):
    v = TfidfVectorizer(tokenizer=tokenize,stop_words='english',strip_accents='unicode', lowercase=True)
    return v.fit(X) ## fit is returning vecorizer

def vectorize_trans(v, X):
    vector = v.transform(X) ## removed nparray()
    words = v.get_feature_names()
    top_features(v,words,10)
    return vector

def clf_model(X,y):
    classifier = MultinomialNB()
    classifier.fit(X, y)
    return classifier # returns the model

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

def top_features(classifier, words, n):
    #input: classifier, int
        #classifier = tfidf vectorizer
        #n = number of top words to report
    #output: list of words
    indices = np.argsort(classifier.idf_)[::-1]
    [words[i] for i in indices[:10]]
    print "Top words are:"
    for i in indices[:10]:
        print words[i]
# examples = ['hate hate hate', "great fun awsome", "sucks"]


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
#plot_datatwin(pos,neg,len(neg))

if __name__ == '__main__':
    X, y = process_file('/Users/janehillyard/Documents/capstone/hate-speech/data/sentimenttrain.txt')
    X = clean_data(X)
    X_train, X_test, y_train, y_test = train_split(X,y)
    #tfidf,vectorizer = vectorize_fit(X_train)
    print len(X_test), len(y_test)
    test_vec = make_vecs(X_test)
    #mymodel  = clf_model(vectorizer,y_train)
    mymodel  = clf_model(test_vec.transform(X_test),y_test)
    test_vectors = test_vec.transform(X_test) #vectorize_trans(tfidf,X_test)
    print score(mymodel,test_vectors,y_test) # used to be test_vectors

    with open('/Users/janehillyard/Documents/capstone/hate-speech/data/vectorizer.pkl', 'w') as f:
        pickle.dump(test_vec, f)
    with open('/Users/janehillyard/Documents/capstone/hate-speech/data/mymodel.pkl', 'w') as f:
        pickle.dump(mymodel, f)
