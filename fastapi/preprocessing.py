import numpy as np

def get_roberta_data(text,tokenizer,sentiment,max_len=512):
    idx0=-1
    idx1=-1
    text=' '.join(text.split())
    if sentiment is not None:
        sentiment=' '.join(sentiment.split())
        sentiment=' '+sentiment
        senti_id=tokenizer.encode(sentiment).ids
        senti_offset=tokenizer.encode(sentiment).offsets
    text=' '+text
    char_level_list=[0]*len(text)
    token_ids=tokenizer.encode(text).ids
    offsets=tokenizer.encode(text).offsets
    if len(token_ids)>(max_len-5):
        if senti_id is not None:
            token_ids=[0]+token_ids[:(max_len-5)]+[2]+[2]+senti_id+[2]
            offsets=[(-1,-1)]+offsets[:(max_len-5)]+[(-1,-1)]+[(-1,-1)]+senti_offset+[(-1,-1)]
        else:
            token_ids=[0]+token_ids[:(max_len-2)]+[2]
            offsets=[(-1,-1)]+offsets[:(max_len-2)]+[(-1,-1)]
    else:
        if senti_id is not None:
            pad_len=max_len-len(token_ids)-5
            token_ids=[0]+token_ids+[2]+[2]+senti_id+[2]+[1]*pad_len
            offsets=[(-1,-1)]+offsets+[(-1,-1)]+[(-1,-1)]+senti_offset+[(-1,-1)]+[(-1,-1)]*pad_len
        else:
            pad_len=max_len-len(token_ids)-2
            token_ids=[0]+token_ids+[2]+[1]*pad_len
            offsets=[(-1,-1)]+offsets+[(-1,-1)]+[(-1,-1)]*pad_len
    attention_mask=np.not_equal(1,token_ids).astype('int')
    return {'input_ids':token_ids,'attention_mask':attention_mask}

def get_prediction_ids(p1,p2,p3,p4,p5):
    start_ids=(p1['start_ids']+p2['start_ids']+p3['start_ids']+p4['start_ids']+p5['start_ids'])/5.0
    end_ids=(p1['end_ids']+p2['end_ids']+p3['end_ids']+p4['end_ids']+p5['end_ids'])/5.0
    start_id=np.argmax(start_ids,axis=-1)[0]
    end_id=np.argmax(end_ids,axis=-1)[0]
    return start_id,end_id

def get_prediction_roberta_string(prediction_ids,tokenizer):
    prediction_tokens=tokenizer.decode(prediction_ids)
    prediction=''
    for i,tok in enumerate(prediction_tokens):
        if tok.startswith(f"{tokenizer.encode(' d').tokens[0][0]}") and i==0:
            prediction+=tok.replace(tokenizer.encode(' d').tokens[0][0],'')
        elif tok.startswith(f"{tokenizer.encode(' d').tokens[0][0]}") and (i!=0):
            prediction=prediction+' '+tok.replace(tokenizer.encode(' d').tokens[0][0],'')
        else:
            prediction+=tok
    return prediction