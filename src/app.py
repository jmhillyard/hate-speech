from flask import Flask, request,render_template
from collections import Counter
import cPickle as pickle
import pandas as pd
import numpy as np
import cap_buildmodel as cb
import mymodel as mm
import mongowork as mw
import clean_data as cld
from nltk.stem import SnowballStemmer
import nltk
import json
app = Flask(__name__)

def list_to_html(d):
    return '<br>'.join('{0}'.format(k, d[k]) for k in d)

# home page
@app.route('/')
def index():
    # Opening screen with graph and feature words
    # setup database connection to mongo db
    dbconn = mw.MongoWork()
    mymod = mm.MyModel()
    # part 1 process new file if exists
    clean = cld.CleanData()
    return_code = ''
    #print ls '/Users/janehillyard/Documents/capstone/hate-speech/src/output_2016*.json'
    filename = '/Users/janehillyard/Documents/capstone/hate-speech/src/output.json'

    text,return_code = clean.convert_tweets(filename)
    if return_code == 0:
        text = np.array(clean.clean_data(text))
        vec = vectorizer.transform(text) #vectorizer.transform(text)
        words = vectorizer.get_feature_names()
        top_f = cb.top_features(vectorizer,words,10)
        pos,neg = np.array(mymod.pred(model,vec))
        print 'this is pos or neg',pos,neg
        dbconn.load_results(pos,neg)
        #indices = np.argsort(vectorizer.idf_)[::-1]
        #print indices[2], words[2]
        print vectorizer.idf_[::-1]
        pred = mymod.predict(model,vec)
        neg_words = mymod.get_neg_features(words,pred)
        print neg_words

        # clean.process_file(filename)
    else:
        top_f = 'NONE at this time.'
    # page = 'Section name prediction.<br><br>pos prediction: {0} <br>pos prediction: {1} <br>  Top ten words {2}'
    # return page.format(pos, neg, top_f)
    #return render_template('welcome.html', data = data)

    # get database values for graph
    # json list sent to HTML for graph
    a = dbconn.get_results()
    lst = json.dumps(a)
    # page = 'Section name prediction.<br><br>pos prediction: {0} <br>pos prediction: {1} <br>  Top ten words {2}'
    # return page.format(pos, neg, top_f)
    # print top_f
    return render_template('welcome.html', vars=lst, top=top_f )

# #
# #
# # # My word counter app
# @app.route('/predict') # methods=['POST'] )
# def predict():

    # # text = filein("../data/sentimenttrain.txt")
    # # text = str(cb.clean_data(text))
    # clean = cld.CleanData()
    # clean.convert_tweets('output.json')
    # text = str(clean.clean_data())
    # #text = str(request.form['user_input'])
    # # do model.predict(vecorizer(text))
    # #vec = vectorizer.transform([text])
    # vec = vectorizer.transform(text)
    # words = vectorizer.get_feature_names()
    # top_f = cb.top_features(vectorizer,words,10)
    # pos,neg = np.array(mm.pred(model,vec))
    #
    # #mpld3.show()
    # data = dict({110:40, 200:300})
    # print type(data)
    # page = 'Section name prediction.<br><br>pos prediction: {0} <br>pos prediction: {1} <br>  Top ten words {2}'
    # return page.format(pos, neg, top_f)
    #return render_template('welcome.html', data = data)
# home page
@app.route('/WordFeatures')

def stem_tokens(tokens,stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens,stemmer)
    return stems

def filein(filename):
    text=[]
    with open(filename) as f:
        lines = f.readlines()
        for i in lines:
            text.append(i[2:])
    return text

if __name__ == '__main__':
    global vectorizer
    global model
    stemmer = SnowballStemmer('english')
    with open('../data/vectorizer.pkl') as f:
        vectorizer = pickle.load(f)
    with open('../data/mymodel.pkl') as f:
        model = pickle.load(f)
    app.run(host='0.0.0.0', port=8010, debug=True)
