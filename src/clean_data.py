import pandas as pd
import numpy as np
import re
import os
import os.path
import yaml
import json
from time import gmtime, strftime
from nltk.corpus import wordnet

class CleanData(object):
    def __init__(self, model=None):
        self.X = None

    def clean_data(self,X):
        # Function: lowercase (also completed in vecorizer),
        #           remove special characters, remove numbers
        # Input: numpy array of strings of text
        # Output:  numpy string of cleansed text
        #text = [re.sub(r'[^\w\s\d]','',h.lower()) for h in X]
        l2=[]
        l=[]
        regex = re.compile('[^a-zA-Z]')
        for i in X:

            text = " ".join(filter(lambda x:x[0]!='@', i.split()))
            text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
            text = re.sub(r'[^\w\s\d]','',text.lower())
            l.append(regex.sub(' ', str(text)))
            # for word in line.split():
            #     if wordnet.synsets(word):
            #         l.append(word)
            # if text != '':
            #     l.append(str(text))
        return np.array(l)


    def convert_tweets(self, filename):
        # read the entire file into a python array
        # Function:
        # Input: file name fully qualified
        # Output: Converted tweets ready to process OR
        #         Message file does not exist
        self.X=[]
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            with open(filename, 'rb') as f:
                data = f.readlines()
        else:
            print "Either file is missing or is not readable"
            return np.array('None'),'U099'
        # remove the trailing "\n" from each line
        data = map(lambda x: x.rstrip(), data)
        data_json_str = "[" + ','.join(data) + "]"
        # now, load it into pandas
        df = pd.read_json(data_json_str)
        df = df['text']
        self.X = np.array(df.values)
        return self.X, 0

    def clean_tweets(self,X):
        # Function: lowercase (also completed in vecorizer),
        #           remove special characters, remove numbers
        # Input: numpy array of strings of text
        # Output:  numpy string of cleansed text
        #text = [re.sub(r'[^\w\s\d]','',h.lower()) for h in X]
        # clean tweets no NUMBERS
        l=[]
        l2=[]
        regex = re.compile('[^a-zA-Z]')
        for i in X:
            text = " ".join(filter(lambda x:x[0]!='@', i.split()))
            text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
            text = re.sub(r'[^\w\s\d]','',text.lower())
            line = str(l)
            l=[]
            for word in line.split(): # get only english words
                if wordnet.synsets(word):
                    l.append(word)
            if text != '':   # remove empty rows
                l.append(str(text))
            l2.append(' '.join(l))
        return np.array(l2)


    def process_file(self,filename):
        modifiedTime = os.path.getmtime(filename)
        timestamp = strftime("%Y-%m-%d%H:%M:%S", gmtime()) #datetime.fromtimestamp(modifiedTime).strftime("%b-%d-%Y_%H.%M.%S")
        prevName = filename
        newName = '/Users/janehillyard/capstone/hate-speech/data/output'
        print newName
        os.rename(filename, newName+"_"+timestamp + ".json")
