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


    def load_results(self,pos,neg,duration=10,dbschool='PEA'):
        # Function:
        # Input:
        # Output:
        data = [{"school":school,'positive': pos, 'negative': neg, 'duration': duration}]
        n = json.dumps(data)
        self.db.results_t.insert(data)


    def get_results(self):
        # Function:
        # Input:
        # Output:
        #print self.db['results_t'].find({},{'school':1, 'positive':1, 'negative':1, '_id':0})
        tlist = [(jsonobj) for jsonobj in self.db['results_t'].find({},{'school':1, 'positive':1, 'negative':1, '_id':0})]
        print 'in results', tlist[0]
        print json.dumps(tlist)

        return json.dumps(tlist)


if __name__ == '__main__':
    print 'in main'
    mymongo = MongoWork()
    mymongo.connect_to_db()
    data = mymongo.get_results()
    jdata = json.loads(data)
    labels = ['Instance', 'Positive', 'Negative']
    pos = [li['positive'] for li in jdata]
    neg = [li['negative'] for li in jdata]
    arr = np.array(labels)
    print data

    print jdata
    i = 0
    for p in pos:
        for n in neg:
            i  += 1
            newrow = [i,p,n]
            arr = np.append(arr,newrow)
    # this is json
    print data

    # ['Year', 'Sales', 'Expenses'],
    #       ['2004',  1000,      400],
    #       ['2005',  1170,      460],
    #       ['2006',  660,       1120],
    #       ['2007',  1030,      540]
