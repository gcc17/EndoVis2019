import json

all_params = json.load(open('config.json'))
locals().update(all_params)

training_params = training_params[model_type]