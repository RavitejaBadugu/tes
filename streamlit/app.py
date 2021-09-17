import streamlit as st
import requests

st.title('welcome to the app!')

st.header('enter the text to find the part of text responsible for the sentiment of text')

tweet=st.text_input('enter the text')
sentiment_maps={0:'neutral',1:'positive',2:'negative'}
if tweet:
    with st.spinner('prediction is in progress'):
        tweet_data={'tweet':tweet}
        extraction_data={'tweet':tweet,'sentiment':0}
        response=requests.post('http://fastapi:8000/sentiment',json=tweet_data).json()
        extraction_data['sentiment']=sentiment_maps.get(response['sentiment'])
        extracted_pred=requests.post('http://fastapi:8000/extraction',json=extraction_data).json()
        st.success(f"the sentiment given text lead is {extraction_data['sentiment']}")
        st.success(f"part of text responsible for that sentiment is {extracted_pred['extracted_text']}")
    