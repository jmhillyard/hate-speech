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

    # part 1 process new file if exists
    clean = cld.CleanData()
    return_code = ''
    filename = '/Users/janehillyard/Documents/capstone/hate-speech/src/output.json'
    return_code = clean.convert_tweets(filename)
    print return_code
    if return_code == None:
        text = str(clean.clean_data())
        #text = str(request.form['user_input'])
        # do model.predict(vecorizer(text))
        #vec = vectorizer.transform([text])
        vec = vectorizer.transform(text)
        words = vectorizer.get_feature_names()
        top_f = cb.top_features(vectorizer,words,10)
        pos,neg = np.array(mm.pred(model,vec))
        dbconn.load_results(pos,neg)
        data = dict({110:40, 200:300})
        print 'I am in app.py', pos,neg, data, top_f
        clean.process_file(filename)
    # page = 'Section name prediction.<br><br>pos prediction: {0} <br>pos prediction: {1} <br>  Top ten words {2}'
    # return page.format(pos, neg, top_f)
    #return render_template('welcome.html', data = data)

    # get database values for graph
    # json list sent to HTML for graph
    a = dbconn.get_results()
    lst = json.dumps(a)
    return render_template('welcome.html', vars=lst)

# #
# #
# # # My word counter app
# @app.route('/predict') # methods=['POST'] )
# def predict():

    # text = filein("../data/sentimenttrain.txt")
    # text = str(cb.clean_data(text))
    clean = cld.CleanData()
    clean.convert_tweets('output.json')
    text = str(clean.clean_data())
    #text = str(request.form['user_input'])
    # do model.predict(vecorizer(text))
    #vec = vectorizer.transform([text])
    vec = vectorizer.transform(text)
    words = vectorizer.get_feature_names()
    top_f = cb.top_features(vectorizer,words,10)
    pos,neg = np.array(mm.pred(model,vec))

    #mpld3.show()
    data = dict({110:40, 200:300})
    print type(data)
    # page = 'Section name prediction.<br><br>pos prediction: {0} <br>pos prediction: {1} <br>  Top ten words {2}'
    # return page.format(pos, neg, top_f)
    #return render_template('welcome.html', data = data)


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
    with open('../data/vectorizer.pkl') as f:
        vectorizer = pickle.load(f)
    with open('../data/mymodel.pkl') as f:
        model = pickle.load(f)
    app.run(host='0.0.0.0', port=8010, debug=True)
