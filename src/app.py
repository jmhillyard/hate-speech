from flask import Flask, request,render_template
from collections import Counter
import cPickle as pickle
import pandas as pd
import numpy as np
import cap_buildmodel as cb
import mymodel as mm
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
    import json
    a=[["Year","Sales","Expenses"],
         [2004,1000,400],
         [2005,1170,460],
         [2006,660,1120],
         [2007,1030,540]]
    cols = [["Year2","Sales2","Expenses2"]]
    print 'these are the cols',cols
    lst = json.dumps(a)
    print lst
    return render_template('welcome.html', vars=lst, colstest=cols)

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
    import json

    # # ... create matplotlib figure
    # fig = cb.plot_datatwin(pos,neg,6)
    # json01 = json.dumps(mpld3.fig_to_dict(fig))
    import matplotlib.pyplot as plt, mpld3
    x = np.linspace(0,6,6)
    # plt.plot(x, neg,x, pos)
    # plt.plot(x, reg_pos, x, reg_neg)
    total = []
    for i in range(0,len(pos)):
        total.append((pos[i] + neg[i]))
    fig, ax1 = plt.subplots()
    print 'before plot', x, pos
    ax1.plot(x, pos, 'b-')
    ax1.plot(x, neg, 'r-')
    ax1.plot(x, total, 'y-')
    ax1.set_xlabel('Unit of Time')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Number of Tweets', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    type(top_f)

    # js_data = json.dumps(mpld3.fig_to_dict(fig))
    # return render_to_response('plot.html', {"my_data": js_data})


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
