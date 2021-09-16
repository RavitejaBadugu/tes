from fastapi import FastAPI
from pydantic import BaseModel
from utils import *
app=FastAPI()


class Input_tweet(BaseModel):
    tweet: str

class Extraction_data(BaseModel):
    tweet: str
    sentiment: str

@app.post('/sentiment')
def get_sentiment(sentence: Input_tweet):
    data=sentence.dict()
    sentiment_predicted=get_sentiment_predict(data['tweet'])
    return {'sentiment':sentiment_predicted}

@app.post('/extraction')
def get_extraction(input_data: Extraction_data):
    data=input_data.dict()
    extracted_text=get_extracted_text(data['tweet'],data['sentiment'])
    return {'extracted_text': extracted_text}