import json
import warnings, random
warnings.simplefilter('ignore')

import pandas as pd

from utils import *

train = pd.read_csv('./input/PG_S03E06/train_df.csv')
test = pd.read_csv('./input/PG_S03E06/test_df.csv')

with open('./output/Hypertuning/230215124817/xgb_params_best.json', 'r') as f:
    xgb_params_best = json.load(f)

xgb_config = {
    'xgb_params': xgb_params_best,
    'features': [feature for feature in train.columns if feature not in ['id', target_name, label_name]],
    'folds':5,
    'seed': 2023
}

XGBR_train_and_predict(train, test, xgb_config, aug=None, run_id=None)