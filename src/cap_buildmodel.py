from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
import random
from nltk.stem import SnowballStemmer,WordNetLemmatizer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import cPickle as pickle
import clean_data as cld
import mymodel as mm

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
        random.shuffle(lines)
        for i in lines:
            label.append(i[0])
            text.append(i[1:])
    X = np.array(text)
    y = np.array(label)
    return X,y

# def clean_data(X):
#     # Function: lowercase (also completed in vecorizer),
#     #           remove special characters, remove numbers
#     # Input: numpy array of strings of text
#     # Output:  numpy string of cleansed text
#     #text = [re.sub(r'[^\w\s\d]','',h.lower()) for h in X]
#     l=[]
#     regex = re.compile('[^a-zA-Z0-9]')
#     for i in X:
#         text = " ".join(filter(lambda x:x[0]!='@', i.split()))
#         text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
#         text = re.sub(r'[^\w\s\d]','',text.lower())
#         l.append(regex.sub(' ', str(text)))
#     return np.array(l)


def do_stemming(X):
    stemmed = []
    for item in X:
        stemmed.append(SnowballStemmer('english').stem(item))
    return stemmed

def stem_tokens(tokens,stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def lem_tokens(tokens,stemmer):
    lem=[]
    for item in tokens:
        lem.append(lemma.lemmatize(item))
    return lem

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = stem_tokens(tokens,stemmer)
    tokens = lem_tokens(tokens,lemma)
    return tokens

def train_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def make_vecs(X):  #tokenizer=tokenize

    v = TfidfVectorizer(tokenizer=tokenize,stop_words='english',
         lowercase=True)#, ngram_range=(1,3))min_df=0, max_df=1)
    v_fit = v.fit(X)
    return v_fit ## fit is returning vecorizer

def clf_model(X,y,alpha=1):
    classifier = MultinomialNB(alpha)
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


def predict(model,X):
    #input classifier  and text to predict
    #Return: prediction
    predictions= model.predict(X)
    return predictions

def top_features(vec, words, n):
    #input: classifier, int
        #classifier = tfidf vectorizer
        #n = number of top words to report
    #output: list of words
    indices = np.argsort(vec.idf_)[::-1]
    return [words[i] for i in indices[:n]]
    # print "Top words are:"
    # for i in indices[:10]:
    #     print words[i]
# examples = ['hate hate hate', "great fun awsome", "sucks"]


#plot_roc(v_probs, y_test,"ROC plot of churn data","False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity, Recall)")
#plot_datatwin(pos,neg,len(neg))

if __name__ == '__main__':
    mymod = mm.MyModel()
    clean = cld.CleanData()
    stemmer = SnowballStemmer('english')
    lemma = WordNetLemmatizer()

    #split file]#old_tweets.txt .86 roc
    # sancsv2 roc.99
    # sentiment rox 0
    X, y = process_file('/Users/janehillyard/capstone/hate-speech/data/gold_tweets.txt')
    X = clean.clean_data(X)  # this is taking a bit of time
    X_train, X_test, y_train, y_test = train_split(X,y)
    train_vec = make_vecs(X_train) # from test
    mymodel  = clf_model(train_vec.transform(X_train),y_train,.01)
    test_vectors = train_vec.transform(X_test) #vectorize_trans(tfidf,X_test)


    ### Model Scoring


    print "model score = ", score(mymodel,test_vectors,y_test) # used to be test_vectors
    pred = predict(mymodel,test_vectors)
    words = train_vec.get_feature_names()
    print "top features", top_features(train_vec,words, 10)
    #print "negative features", mymod.get_neg_features(words,y_train)
    with open('/Users/janehillyard/capstone/hate-speech/data/vectorizer.pkl', 'w') as f:
        pickle.dump(train_vec, f)
    with open('/Users/janehillyard/capstone/hate-speech/data/mymodel.pkl', 'w') as f:
        pickle.dump(mymodel, f)

    ### get most frequent words
    probs = mymodel.predict_proba(test_vectors)
    frequent_words = mymod.get_doc_frequencies(X_test, probs)
    print "top 10 Negative tweets  are: ",frequent_words
    # method I: plt
        #roc curve
    from sklearn.metrics import roc_curve,auc
    import matplotlib.pyplot as plt


    preds = probs[:,1]
    yint = y_test.astype(np.float)
    fpr, tpr, threshold = roc_curve(yint, preds)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
