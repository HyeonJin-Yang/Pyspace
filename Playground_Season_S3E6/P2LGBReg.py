import json
import warnings, random
warnings.simplefilter('ignore')

import pandas as pd

from utils import *

train = pd.read_csv('./input/PG_S03E06/train_dfcsv')
test = pd.read_csv('./input/PG_S03E06/test_df.csv')

with open('./output/Hypertuning/230214123554/lgbm_params_best.json', 'r') as f:
    lgbm_params_best = json.load(f)

lgbm_config = {
    'lgbm_params': lgbm_params_best,
    'features': [feature for feature in train.columns if feature not in ['id', target_name, label_name]],
    'eval_metric': 'rmse',
    'best_iteration': 100,
    'folds':5,
    'seed': 2023
}

LGBR_train_and_predict(train, test, lgbm_config, aug=None, run_id=None)