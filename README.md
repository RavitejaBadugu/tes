This project is close to Question answering task in nlp.

In question answering task given context,question we need to get the answer from the context. In this project we are given text and sentiment of that text 

we need to find part of text lead to that sentiment.

It is a kaggle competition. Link is https://www.kaggle.com/c/tweet-sentiment-extraction
My score for this competition is::
![alt text](https://github.com/RavitejaBadugu/tes/blob/main/tweet_images/Screenshot%202021-12-20%20202239.png)

Models are deployed in ec2 gpu instance using tensorflow serving.

Used libraries::

DVC-> it is used for model tracking and config file tracking.

HuggingFace-> this contains pretrained NLP models. For this task I used Roberta base models.

fastapi-> used for creating backend part for this application.

Streamlit-> used for user interface.

docker-> It is used for packaging the code and maintaining the versions of the libraries.

Tensorflow serving-> Models which are finalized are deployed using tensorflow serving which creates a endpoint. So, models can be deployed anywhere and we can invoke the endpoint

and send the data, in return it gives the output.

Check tf serving locally::
![alt text](https://github.com/RavitejaBadugu/tweet_sentiment_extraction/blob/main/tweet_images/Inkedupdated_serving_check_locally.jpg)

Checking tf serving in ec2::
![alt text](https://github.com/RavitejaBadugu/tweet_sentiment_extraction/blob/main/tweet_images/Inkedupdated_ec2_serving_exraction_LI.jpg)

Given a fake input and checking the output format of the model::
![alt text](https://github.com/RavitejaBadugu/tweet_sentiment_extraction/blob/main/tweet_images/updated_duplicate_input_extractionjpg.jpg)

Docker command for running tf serving in ec2 and follow the same for sentiment models serving::
![alt text](https://github.com/RavitejaBadugu/tweet_sentiment_extraction/blob/main/tweet_images/updated_tf_serving_docker_ec2_cmd.jpg)

Output from the app given a sentence::
![alt text](https://github.com/RavitejaBadugu/tweet_sentiment_extraction/blob/main/tweet_images/positive_test_sample.png)

From above text input if we change "good to negative and add no in front of river" models going to predict that given text is creating negative impact and gives Negative part of sentence as output.
![alt text](https://github.com/RavitejaBadugu/tweet_sentiment_extraction/blob/main/tweet_images/negative_test_sample.png)




