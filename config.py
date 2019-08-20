import json
import numpy as np
import pdb

all_params = json.load(open('config.json')) # To be refactored
locals().update(all_params)

training_params = training_params[model_type]