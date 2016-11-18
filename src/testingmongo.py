from flask import Flask, request
from collections import Counter
import cPickle as pickle
import pandas as pd
import numpy as np
import mymodel as mm
import cap_buildmodel as cb
from nltk.stem import SnowballStemmer
import nltk
import mongowork as mdb
app = Flask(__name__)



def tokenize(text):
    #Tokenize and stemming likely should be elsewhere
    tokens = nltk.word_tokenize(text)
    stems = do_stemming(tokens)
    return stems

def do_stemming(X):
    #Tokenize and stemming likely should be elsewhere
    stemmed = []
    for item in X:
        stemmed.append(SnowballStemmer('english').stem(item))
    return stemmed



if __name__ == '__main__':
    print 'in main'
    mymongo = mdb.MongoWork()
    mymongo.connect_to_db()
    data = mymongo.get_results()
    print data
    # with open('../data/vectorizer.pkl') as f:
    #     vectorizer = pickle.load(f)
    # with open('../data/mymodel.pkl') as f:
    #     model = pickle.load(f)
    #
    #
    # text=[]
    # with open("/Users/janehillyard/Documents/capstone/capstonetrainingdata/sentimenttrain.txt") as f:
    #     lines = f.readlines()
    #     np.random.shuffle(lines)
    #     type(lines)
    #     for i in lines:
    #         text.append(i[2:])
    #
    # X = np.array(text)
    # print X.shape
    # #text = str('happy sad and more and so much more with the future of the  ')
    #
    # # do model.predict(vecorizer(text))
    # #vec = vectorizer.transform([text])
    # vec = vectorizer.transform(X)
    # print vec
    # print type(vec), vec.shape
    # pred = cb.pred(model,vec)
