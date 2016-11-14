from pymongo import MongoClient
import numpy as np
import json

class MongoWork(object):
    def __init__(self, model=None):
        self.db = None


    def connect_to_db(self):
        """
        Connects to database
        NOTE: if database does not exist, one will be created

        Parameters
        ----------
        None

        Returns
        -------
        db = database fraud
        """
        conn = MongoClient('mongodb://localhost:27017/')
        # connect to the students database and the ctec121 collection
        self.db = conn.hatespeech
        return self.db


    def load_results(self,pos,neg,duration=10,school='PEA'):
        # Function:  load pos and neg results
        # Input:  # pos tweets, # neg Tweets
        # Output: none
        m = MongoWork()
        db = m.connect_to_db()
        tot = pos + neg
        db.results_t.insert({"school":school,'positive': pos, 'negative': neg, 'duration': duration, 'total':tot})


    def get_results(self):
        # Function: pulls all rows from mongo db
        # Input: None
        # Output: list of lists with labels
        #print self.db['results_t'].find({},{'school':1, 'positive':1, 'negative':1, '_id':0})
        m = MongoWork()
        db = m.connect_to_db()
        data = [(jsonobj) for jsonobj in db['results_t'].find({},{'school':1, 'positive':1, 'negative':1, 'total':1,'_id':0})]
        pos = [li['positive'] for li in data]
        neg = [li['negative'] for li in data]
        tot = [li['total'] for li in data]
        labels = ['Instance', 'Positive', 'Negative','Total']
        l = []
        l.append(pos)
        l.append(neg)
        l.append(tot)
        l2 = []
        for i in range(len(data)):
            l2.append([i,pos[i],neg[i],tot[i]])
        # this is json
        l2.insert(0,labels)
        return l2
