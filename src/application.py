from flask import Flask, request,render_template
import cPickle as pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer,WordNetLemmatizer
import nltk
import json
import cap_buildmodel as cb
import mymodel as mm
import dbwork as dbw
import clean_data as cld
application = Flask(__name__)
# this is for AWS porting below
# global vectorizer
# global model
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = stem_tokens(tokens,stemmer)
    tokens = lem_tokens(tokens,lemma)
    return tokens
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

stemmer = SnowballStemmer('english')
lemma = WordNetLemmatizer()
with open('../data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('../data/mymodel.pkl') as f:
    model = pickle.load(f)


# this is for AWS porting above

def list_to_html(d):
    return '<br>'.join('{0}'.format(k, d[k]) for k in d)

# home page
@application.route('/')
def index():
    # setup database connection to mongo db or dynamo
    dbconn = dbw.DbWork()
    mymod = mm.MyModel()
    clean = cld.CleanData()
    return_code = 0
    filename = '/Users/janehillyard/capstone/hate-speech/src/output.json'

    text,return_code = clean.convert_tweets(filename)
    #file exists ready to process tweets
    if return_code == 0:
        text = np.array(clean.clean_data(text))
        vec = vectorizer.transform(text) #vectorizer.transform(text)
        words = vectorizer.get_feature_names()
        top_f = cb.top_features(vectorizer,words,10)
        pos,neg = np.array(mymod.pred(model,vec))
        dbconn.load_results(pos,neg)
        pred = mymod.predict(model,vec)

        ### get most frequent neg tweets
        probs = model.predict_proba(vec)
        top_neg_tweets = mymod.get_doc_frequencies(text, probs)
        clean.process_file(filename)
    else:
        top_f = 'There are no new Tweets to Process.'
        top_neg_tweets= ['There are no new Tweets to Process.',0.000]
    # page = 'Section name prediction.<br><br>pos prediction: {0} <br>pos prediction: {1} <br>  Top ten words {2}'
    # return page.format(pos, neg, top_f)
    #return render_template('welcome.html', data = data)

    # get database values for graph
    # json list sent to HTML for graph
    a = dbconn.get_results()
    lst = json.dumps(a)
    lst_top_neg = json.dumps(top_neg_tweets)

    return render_template('welcome.html', vars=lst, top_tweets=lst_top_neg)


if __name__ == '__main__':
    # global vectorizer
    # global model
    # stemmer = SnowballStemmer('english')
    # lemma = WordNetLemmatizer()
    # with open('../data/vectorizer.pkl') as f:
    #     vectorizer = pickle.load(f)
    # with open('../data/mymodel.pkl') as f:
    #     model = pickle.load(f)
    # this it to run local:
    # application.run(host='0.0.0.0', port=8010, debug=True)
    application.run(host='0.0.0.0', port=8010, debug=True)
