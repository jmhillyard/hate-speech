from pymongo import MongoClient
import numpy as np
import json


import boto3

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
        print 'in mongo ', pos, neg, tot
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


        # TestInput = {'event_created':'3', "school":"pa", "positive":100, "negative":300, "duration":10, "total":400}
        #
        # test_json = json.dumps(TestInput)
        #
        # loaded_test_json = json.loads(test_json)
        #
        # Write_Record('results_t', loaded_test_json)






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




# #This is the line you need to change to reflect your json record.
# TestInput = {'event_created':'1', "school":"pa", "positive":100, "negative":300, "duration":10, "total":400}
#
# test_json = json.dumps(TestInput)
#
# loaded_test_json = json.loads(test_json)
#
# Write_Record('results_t', loaded_test_json)
