from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
from auth import TwitterAuth
from time import gmtime, strftime
import time
import os

  #Very simple (non-production) Twitter stream example
  #1. Download / install python and tweepy (pip install tweepy)
  #2. Fill in information in auth.py
  #3. Run as: python streaming_simple.py
  #4. It will keep running until the user presses ctrl+c to exit
  #All output stored to output.json (one tweet  per line)
  #Text of tweets also printed as recieved (see note about not doing this in production (final) code

class StdOutListener(StreamListener):

	def __init__(self, time_limit=60):
		self.start_time = time.time()
		self.limit = time_limit
		super(StdOutListener, self).__init__()


	#This function gets called every time a new tweet is received on the stream
	def on_data(self, data):
		if (time.time() - self.start_time) < self.limit:

			#Just write data to one line in the file
			fhOut.write(data)

			#Convert the data to a json object (shouldn't do this in production; might slow down and miss tweets)
			j=json.loads(data)

			#See Twitter reference for what fields are included -- https://dev.twitter.com/docs/platform-objects/tweets
			text=j["text"] #The text of the tweet
			#print(text) #Print it out
			## checking to see if I can stop by count

			return True
		else:
			#Just write data to one line in the file
			fhOut.write(data)

			#Convert the data to a json object (shouldn't do this in production; might slow down and miss tweets)
			j=json.loads(data)

			#See Twitter reference for what fields are included -- https://dev.twitter.com/docs/platform-objects/tweets
			text=j["text"] #The text of the tweet
			#print(text) #Print it out
			## checking to see if I can stop by count

			return False


	def on_error(self, status):
		print("ERROR")
		print(status)


if __name__ == '__main__':

	try:

		#Create a file to store output. "a" means append (add on to previous file)
		filename = "output.json"
		if os.path.isfile(filename):
			modifiedTime = os.path.getmtime(filename)
			timestamp = strftime("%Y-%m-%d%H:%M:%S", gmtime()) #datetime.fromtimestamp(modifiedTime).strftime("%b-%d-%Y_%H.%M.%S")
			prevName = filename
			newName = 'output'
			os.rename(prevName, newName+"_"+timestamp + ".json")

		fhOut = open(filename,"a")

		#Create the listener
		l = StdOutListener(time_limit=600)
		auth = OAuthHandler(TwitterAuth.consumer_key, TwitterAuth.consumer_secret)
		auth.set_access_token(TwitterAuth.access_token, TwitterAuth.access_token_secret)

		#Connect to the Twitter stream
		stream = Stream(auth, l)

		#Terms to track
		#stream.filter(track=["fraud"])

		#Alternatively, location box  for geotagged tweets
		# newburyport to rochester, Hampton and Seabrook
		south_nh =[-71,42.4,-70.5,43.3]
		stream.filter(locations=south_nh)

	except KeyboardInterrupt:
		#User pressed ctrl+c -- get ready to exit the program
		pass

	#Close the
	fhOut.close()
