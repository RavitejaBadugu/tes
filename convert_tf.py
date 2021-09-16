from requests import models
import tensorflow as tf
import argparse
from transformers import TFRobertaModel,RobertaConfig
from tensorflow.keras.layers import Input,Dropout,Dense,Conv1D,Flatten,Softmax
from tensorflow.keras.models import Model,load_model

def get_seniment_model():
    tf.keras.backend.clear_session()
    config = RobertaConfig.from_pretrained('roberta_offline/config-roberta-base.json')
    roberta_model = TFRobertaModel.from_pretrained('roberta_offline/pretrained-roberta-base.h5',config=config)
    ins1=Input((110,),dtype=tf.int32,name='input_ids')
    ins2=Input((110,),dtype=tf.int32,name='attention_mask')
    pre_layers=roberta_model({'input_ids':ins1,'attention_mask':ins2})
    drop=Dropout(0.1)(pre_layers[1])
    outs=Dense(3,activation='softmax')(drop)
    model=Model(inputs={'input_ids':ins1,'attention_mask':ins2},
           outputs=outs)
    return model
    

def get_extraction_model():
    tf.keras.backend.clear_session()
    config = RobertaConfig.from_pretrained('roberta_offline/config-roberta-base.json')
    roberta_model = TFRobertaModel.from_pretrained('roberta_offline/pretrained-roberta-base.h5',config=config)
    ins1=Input((110,),dtype=tf.int32,name='input_ids')
    ins2=Input((110,),dtype=tf.int32,name='attention_mask')
    pre_layers=roberta_model({'input_ids':ins1,'attention_mask':ins2})
    drop=Dropout(0.1)(pre_layers[0])
    d=Conv1D(2,1)(drop)
    x1,x2=tf.keras.layers.Lambda(lambda a:tf.split(a,2,axis=-1))(d)
    x1=Flatten()(x1)
    out1=Softmax(name='start_ids')(x1)
    x2=Flatten()(x2)
    out2=Softmax(name='end_ids')(x2)
    model=Model(inputs={'input_ids':ins1,'attention_mask':ins2},
           outputs={'start_ids':out1,'end_ids':out2})
    return model

def convert_to_tf_format(args):
    path=args.export_dir
    print(f"got {path} to save models in tf format")
    for i in range(args.folds):
        if args.type=='sentiment':
            model=get_seniment_model()
            model.load_weights(f"sentiment_models/{args.common_name}_{i}.h5")
            print(f"model_{i} is loaded")
        else:
            model=get_extraction_model()
            model.load_weights(f"extraction_models/aws_models/{args.common_name}_{i}.h5")
            print(f"model_{i} is loaded")
        tf.saved_model.save(model,export_dir=f"{path}_{i}/1/")
        print(f"model_{i} is saved")



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',type=int,help='trained model folds',default=1)
    parser.add_argument('--type',type=str,help='type of model')
    parser.add_argument('--common_name',type=str,help='common name of fold models')
    parser.add_argument('--export_dir',type=str,help='common path name for the models')

    args=parser.parse_args()
    convert_to_tf_format(args)
