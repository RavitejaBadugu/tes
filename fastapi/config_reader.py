import yaml

with open('config.yml','r') as y:
    config_file=yaml.safe_load(y)

senti_url=config_file.get('sentiment_models')
extract_url=config_file.get('extraction_models')
MAX_LENGTH=config_file.get('max_length')
print(senti_url.get('fold_0'))