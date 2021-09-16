import tensorflow as tf
from transformers import TFBertModel,TFRobertaModel,RobertaTokenizer
from config_reader import *
import tokenizers
from preprocessing import *
import requests
import json

roberta_tokenizer=tokenizers.ByteLevelBPETokenizer('./tokenizer/vocab-roberta-base.json','./tokenizer/merges-roberta-base.txt',
                                                  lowercase=True)


def get_sentiment_predict(text):
    model_inputs=get_roberta_data(text,roberta_tokenizer,sentiment=None,max_len=MAX_LENGTH)
    data=json.dumps({"signature_name": "serving_default",
     "instances": [model_inputs]})
    headers={"content-type": "application/json"}
    p0=requests.post(senti_url['fold_0'],data=data,headers=headers)
    print(
        'MODEL IS INVOKED'
    )
    print(json.loads(p0.text))
    print(json.loads(p0.text))
    print(json.loads(p0.text))
    print(json.loads(p0.text))
    print(json.loads(p0.text))
    print(json.loads(p0.text))
    pred_0=json.loads(p0.text)['predictions']
    p1=requests.post(senti_url['fold_1'],data=data,headers=headers)
    pred_1=json.loads(p1.text)['predictions']
    p2=requests.post(senti_url['fold_2'],data=data,headers=headers)
    pred_2=json.loads(p2.text)['predictions']
    p3=requests.post(senti_url['fold_3'],data=data,headers=headers)
    pred_3=json.loads(p3.text)['predictions']
    p4=requests.post(senti_url['fold_4'],data=data,headers=headers)
    pred_4=json.loads(p4.text)['predictions']
    prediction=(pred_0+pred_1+pred_2+pred_3+pred_4)/5.0
    return np.argmax(prediction,axis=-1)[0]
    

def get_extracted_text(text,sentiment):
    if sentiment=='neutral':
        return text
    else:
        model_inputs=get_roberta_data(text,roberta_tokenizer,sentiment=sentiment,max_len=MAX_LENGTH)
        data=json.dumps({"signature_name": "serving_default",
        "instances": [model_inputs]})
        headers={"content-type": "application/json"}
        p0=requests.post(extract_url['fold_0'],data=data,headers=headers)
        pred_0=json.loads(p0.text)['predictions']
        p1=requests.post(extract_url['fold_1'],data=data,headers=headers)
        pred_1=json.loads(p1.text)['predictions']
        p2=requests.post(extract_url['fold_2'],data=data,headers=headers)
        pred_2=json.loads(p2.text)['predictions']
        p3=requests.post(extract_url['fold_3'],data=data,headers=headers)
        pred_3=json.loads(p3.text)['predictions']
        p4=requests.post(extract_url['fold_4'],data=data,headers=headers)
        pred_4=json.loads(p4.text)['predictions']
        #if isinstance(pred_0,list):
        #    pred_0=np.asarray(pred_0)
        #    pred_1=np.asarray(pred_1)
        #    pred_2=np.asarray(pred_2)
        #    pred_3=np.asarray(pred_3)
        #    pred_4=np.asarray(pred_4)
        start_id,end_id=get_prediction_ids(pred_0+pred_1+pred_2+pred_3+pred_4)
        if start_id>end_id:
            prediction=text
        else:
            impact=model_inputs['input_ids'][start_id:end_id+1]
            selected_text=get_prediction_roberta_string(impact,roberta_tokenizer)
            if (selected_text==' ') or (selected_text=='') or (selected_text is None):
                prediction=text
            else:
                prediction=selected_text
        return prediction