from pymongo import MongoClient
import numpy as np
import json
import boto3

class DbWork(object):
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
        m = DbWork()
        db = m.connect_to_db()
        tot = pos + neg
        db.results_t.insert({"school":school,'positive': pos, 'negative': neg, 'duration': duration, 'total':tot})
        # adding DYNAMO load
        # # get most recent object nd write_ob
        # oldid = get_records('objid_control_t')
        # print oldid[1]
        # # oldid = [li['objID'] for li in oldid]
        # # print int(oldid)
        # # oldid = int(oldid) + 1
        # print oldid
        # write_obj('objid_control_t', {'objID':'1'})

    def get_results(self):
        # Function: pulls all rows from mongo db
        # Input: None
        # Output: list of lists with labels

        m = DbWork()
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


# # Get the service resource.
# dynamodb = boto3.resource('dynamodb')

def Write_Record(TableName, JSON):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TableName)
    print 'write record to dynamo', table
    print JSON
    table.put_item(
       Item={

           'ObjID': JSON['event_created'],
           'school': JSON['school'],
           'positive': JSON['positive'],
           'negative': JSON['negative'],
           'duration': JSON['duration'],
           'total': JSON['total'],
       }

    )

def write_obj(TableName, JSON):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TableName)
    print 'write record to dynamo', table
    print JSON
    table.put_item(
       Item={

           'objID': JSON['objID']
       }

    )

def get_records(table):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table)
    response = table.scan()
    return response
