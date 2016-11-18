import json
import boto3

# Get the service resource.
dynamodb = boto3.resource('dynamodb')

def Write_Record(TableName, JSON):

   table = dynamodb.Table(TableName)
   print table
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


#This is the line you need to change to reflect your json record.
TestInput = {'event_created':'1', "school":"pa", "positive":100, "negative":300, "duration":10, "total":400}

test_json = json.dumps(TestInput)

loaded_test_json = json.loads(test_json)

Write_Record('results_t', loaded_test_json)
