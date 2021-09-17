This project is close to Question answering task in nlp.
In question answering task given context,question we need to get the answer from the context. In this project we are given text and sentiment of that text 
we need to find part of text lead to that sentiment.
It is a kaggle competition. Link is https://www.kaggle.com/c/tweet-sentiment-extraction
training is done in kaggle.
Models are deployed in ec2 gpu instance using tensorflow serving.
Used libraries::
DVC-> it is used for model tracking and config file tracking.
HuggingFace-> this contains pretrained NLP models. For this task I used Roberta base models.
fastapi-> used for creating backend part for this application.
Streamlit-> used for user interface.
docker-> It is used for packaging the code and maintaining the versions of the libraries.
Tensorflow serving-> Models which are finalized are deployed using tensorflow serving which creates a endpoint. So, models can be deployed anywhere and we can invoke the endpoint
and send the data, in return it gives the output.



