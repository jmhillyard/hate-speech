import pandas as pd
import numpy as np
import cap_buildmodel as cb
import mymodel as mm
import mongowork as mw
import clean_data as cld
import json
import cPickle as pickle
import nltk
from nltk.stem import SnowballStemmer

def stem_tokens(tokens,stemmer):
    stemmed = []
    # print tokens[:10]
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # print 'in tokenize', text[:0]
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens,stemmer)
    # print stems[:10]
    return stems



with open('../data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('../data/mymodel.pkl') as f:
	model = pickle.load(f)

stemmer = SnowballStemmer('english')# part 1 process new file if exists

dbconn = mw.MongoWork()

clean = cld.CleanData()
return_code = ''
filename = '/Users/janehillyard/Documents/capstone/hate-speech/src/output.json'
return_code = clean.convert_tweets(filename)
if return_code == None:
	text = str(clean.clean_data())
	# print 'in app',len(text)
	vec = vectorizer.transform(text)
	print 'before words'
	words = vectorizer.get_feature_names()

	top_f = cb.top_features(vectorizer,words,10)
	pos,neg = np.array(mm.pred(model,vec))
	dbconn.load_results(pos,neg)
	print 'end of db conn'
print 'end of process tweets to db'
print top_f
