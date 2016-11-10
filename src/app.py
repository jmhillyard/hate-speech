from flask import Flask, request
##from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from collections import Counter
import cPickle as pickle
import pandas as pd
import cap_buildmodel as mm
from nltk.stem import SnowballStemmer
import nltk
app = Flask(__name__)

def list_to_html(d):
    return '<br>'.join('{0}'.format(k, d[k]) for k in d)

# home page
@app.route('/')
def submission_page():
    return '''
        <form action="/predict" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''
# My word counter app
@app.route('/predict', methods=['POST'] )
def predict():

    text = str(request.form['user_input'])

    # do model.predict(vecorizer(text))
    #vec = vectorizer.transform([text])
    vec = vectorizer.transform([text])
    pred = model.predict(vec)

    page = 'Section name prediction.<br><br>prediction:<br> {0}'
    return page.format(pred[0])

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
    global vectorizer
    global model
    with open('../data/vectorizer.pkl') as f:
        vectorizer = pickle.load(f)
    with open('../data/mymodel.pkl') as f:
            model = pickle.load(f)
    app.run(host='0.0.0.0', port=8010, debug=True)
