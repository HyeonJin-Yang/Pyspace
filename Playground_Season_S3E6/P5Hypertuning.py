import json
import warnings, os, datetime
warnings.simplefilter('ignore')

import optuna

from utils import *

train = pd.read_csv('./input/PG_S03E06/train_df.csv')
test = pd.read_csv('./input/PG_S03E06/test_df.csv')

xgb_opt_config = {
    'folds': 5,
    'features': [feature for feature in train.columns if feature not in ['id', target_name, label_name]]
}

lgbm_opt_config = {
    'folds': 5,
    'features': [feature for feature in train.columns if feature not in ['id', target_name, label_name]],
    'best_iteration': 10
}

output_path = './output/Hypertuning/' + datetime.datetime.now().strftime('%y%m%d%H%M%S')
if not os.path.exists(output_path):
    os.mkdir(output_path)

n_trials = 100

study_xgb = optuna.create_study(study_name='xgb_parameter_opt',
                                direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=48))    
func_for_xgb = lambda trial: objective(trial, train, xgb_opt_config, 'xgb')
study_xgb.optimize(func_for_xgb, n_trials=n_trials)
with open(output_path + '/xgb_params_best.json', 'w') as f:
    json.dump(study_xgb.best_params, f)

study_lgbm = optuna.create_study(study_name='lgbm_parameter_opt',
                                direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=67))
func_for_lgbm = lambda trial: objective(trial, train, lgbm_opt_config, 'lgbm')
study_lgbm.optimize(func_for_lgbm, n_trials=n_trials)
with open(output_path + '/lgbm_params_best.json', 'w') as f:
    json.dump(study_lgbm.best_params, f)
