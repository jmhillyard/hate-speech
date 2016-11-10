import os

class TwitterAuth:

    consumer_key = os.environ['TW_ACCESS_KEY_ID']
    consumer_secret = os.environ['TW_SECRET_ACCESS_KEY']
    access_token = os.environ['TW_SECRET_TOKEN_KEY_ID']
    access_token_secret = os.environ['TW_SECRET_TOKEN_KEY']
