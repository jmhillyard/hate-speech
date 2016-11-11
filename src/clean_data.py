import pandas as pd
import numpy as np
import re

class CleanData(object):
    def __init__(self, model=None):
        self.X = None

    def clean_data(self):
        # Function: lowercase (also completed in vecorizer),
        #           remove special characters, remove numbers
        # Input: numpy array of strings of text
        # Output:  numpy string of cleansed text

        text = [re.sub(r'[^\w\s\d]','',h.lower()) for h in self.X]
        regex = re.compile('[^a-zA-Z]')
        l=[]
        for i in text:
            l.append(regex.sub(' ', i))
        return np.array(l)

    def convert_tweets(self, filename):
        # read the entire file into a python array
        with open(filename, 'rb') as f:
            data = f.readlines()

        # remove the trailing "\n" from each line
        data = map(lambda x: x.rstrip(), data)
        data_json_str = "[" + ','.join(data) + "]"
        # now, load it into pandas
        df = pd.read_json(data_json_str)
        df = df['text']
        self.X = df
