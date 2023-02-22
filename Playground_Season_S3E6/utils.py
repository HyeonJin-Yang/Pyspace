import os, time, random
import datetime

import numpy as np
import pandas as pd

import optuna
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

idx_name = 'id'
target_name = 'price'
label_name = 'split'

def Construct_model_output_path(model_name, output_root, run_id=None):
    if not run_id:
        run_id = f'run_{model_name}_' + datetime.datetime.now().strftime('%y%m%d%H%M%S')
        time.sleep(1)
        output_path = output_root + run_id + '/'
    else:
        output_path = output_root + run_id + '/'
    return output_path

def Write_Log(logFile, text):
    logFile.write(text)
    logFile.write('\n')
    return None

def LGBR_train_and_predict(train, test, config, output_root='./output/', aug=None, run_id=None):
    # Construct output path and Save Model
    output_path = Construct_model_output_path('LGBR', output_root, run_id)
    if not os.path.exists(output_path): os.mkdir(output_path)
    # os.system(f'cp ./*.py {output_path}') 
    # os.system(f'cp ./*.sh {output_path}') 
    
    # Make log 
    log = open(output_path + 'train.log', 'w', buffering=1)
    log.write(str(config)+'\n')

    # Parameters and Containers
    lgbm_params = config['lgbm_params']
    best_iteration = config['best_iteration']
    features = config['features']
    folds = config['folds']
    seed = config['seed']
    eval_metric = config['eval_metric']

    oof, test_preds = train[[idx_name]], np.zeros(test.shape[0])
    oof[target_name] = 0
    all_scores, feature_importance = [], [] 
    
    # SKF + LGBMRegressor modeling
    cnt = 0
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for i, (trn_idx, val_idx) in enumerate(kf.split(train[features], train[target_name])):
        # Split train data
        X_train, X_val = train.loc[trn_idx, features], train.loc[val_idx, features]
        y_train, y_val = train.loc[trn_idx, target_name], train.loc[val_idx, target_name]

        # Generate, Train and Save model
        model = LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric=eval_metric,
                  verbose=-1)
        model.booster_.save_model(output_path + 'fold%s.ckpt'%i)

        # Calculate score
        y_preds = model.predict(X_val, num_iteration=best_iteration)
        score = metrics.mean_squared_error(y_val, y_preds, squared=False)
        all_scores.append(score)
        oof.loc[val_idx, target_name] = y_preds
        Write_Log(log, 'fold%s valid rmse score is %.6f'%(i, score) )

        # Featrue importances
        importance = model.feature_importances_
        feature_name = model.feature_name_
        feature_importance.append(pd.DataFrame({'feature_name': feature_name,
                                                'importance': importance}))

        # Predict test data
        test_preds += model.predict(test[features], num_iteration=best_iteration)
        cnt += 1

    # Save oof
    oof.to_csv(output_path + 'oof.csv', index=False)

    # Calculate and Save feature importance
    feature_importance_df = pd.concat(feature_importance, axis=0)
    feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
    feature_importance_df = feature_importance_df.sort_values(by=['importance'], ascending=False)
    feature_importance_df.to_csv(output_path + 'feature_importances.csv', index=False)

    # Calculate total score and Make a submisstion
    lgb_score = np.mean(all_scores)
    Write_Log(log, 'All valid rmse score is %.6f'%lgb_score)
    log.close()
    
    lgb_preds = test_preds / cnt
    sub = test[[idx_name]]
    sub[target_name] = lgb_preds
    sub.to_csv(output_path + 'submission.csv', index=False)

    return lgb_score, lgb_preds

def XGBR_train_and_predict(train, test, config, output_root='./output/', aug=None, run_id=None):
    # Construct output path and Save Model
    output_path = Construct_model_output_path('XGBR', output_root, run_id)
    if not os.path.exists(output_path): os.mkdir(output_path)
    # os.system(f'cp ./*.py {output_path}') 
    # os.system(f'cp ./*.sh {output_path}') 
    
    # Make log 
    log = open(output_path + 'train.log', 'w', buffering=1)
    log.write(str(config)+'\n')

    # Parameters and Containers
    xgb_params = config['xgb_params']
    features = config['features']
    folds = config['folds']
    seed = config['seed']

    oof, test_preds = train[[idx_name]], np.zeros(test.shape[0])
    oof[target_name] = 0
    all_scores, feature_importance = [], [] 
    
    # SKF + XGBRegressor modeling
    cnt = 0
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for i, (trn_idx, val_idx) in enumerate(kf.split(train[features], train[target_name])):
        # Split train data
        X_train, X_val = train.loc[trn_idx, features], train.loc[val_idx, features]
        y_train, y_val = train.loc[trn_idx, target_name], train.loc[val_idx, target_name]

        # Generate, Train and Save model
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        model.save_model(output_path + 'fold%s.ckpt'%i)

        # Calculate score
        y_preds = model.predict(X_val)
        score = metrics.mean_squared_error(y_val, y_preds, squared=False)
        all_scores.append(score)
        oof.loc[val_idx, target_name] = y_preds
        Write_Log(log, 'fold%s valid rmse score is %.6f'%(i, score) )

        # Featrue importances
        importance = model.feature_importances_
        feature_name = model.feature_names_in_
        feature_importance.append(pd.DataFrame({'feature_name': feature_name,
                                                'importance': importance}))

        # Predict test data
        test_preds += model.predict(test[features])
        cnt += 1

    # Save oof
    oof.to_csv(output_path + 'oof.csv', index=False)

    # Calculate and Save feature importance
    feature_importance_df = pd.concat(feature_importance, axis=0)
    feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
    feature_importance_df = feature_importance_df.sort_values(by=['importance'], ascending=False)
    feature_importance_df.to_csv(output_path + 'feature_importances.csv', index=False)

    # Calculate total score and Make a submisstion
    xgb_score = np.mean(all_scores)
    Write_Log(log, 'All valid rmse score is %.6f'%xgb_score)
    log.close()
    
    xgb_preds = test_preds / cnt
    sub = test[[idx_name]]
    sub[target_name] = xgb_preds
    sub.to_csv(output_path + 'submission.csv', index=False)

    return xgb_score, xgb_preds

def objective(trial, train, config, model_type):
    folds = config['folds']
    features = config['features']

    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=59)

    if model_type == "xgb":
        xgb_params_optuna = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'max_leaves': trial.suggest_int('max_leaves', 10, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'verbosity': 0,
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.01, 0.5),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1.0),
            'random_state': 2023,
            'eval_metric': metrics.mean_squared_error,
            'early_stopping_rounds': 100,
            'n_jobs': -1
        }

        all_scores = []
        for i, (train_idx, val_idx) in enumerate(split.split(train[features], train[label_name])):
            X_train, X_val = train.loc[train_idx, features], train.loc[val_idx, features]
            y_train, y_val = train.loc[train_idx, target_name], train.loc[val_idx,target_name]

            model = XGBRegressor(**xgb_params_optuna)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            y_preds = model.predict(X_val)

            RMSE = metrics.mean_squared_error(y_val, y_preds, squared=False)
            all_scores.append(RMSE)
    
    if model_type == "lgbm":
        best_iteration = config['best_iteration']
        lgbm_params_optuna = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 500),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'objective': 'regression',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.01, 0.5),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'random_state': 2023,
            'n_jobs': -1
        }

        all_scores = []
        for i, (train_idx, val_idx) in enumerate(split.split(train[features], train[label_name])):
            X_train, X_val = train.loc[train_idx, features], train.loc[val_idx, features]
            y_train, y_val = train.loc[train_idx, target_name], train.loc[val_idx,target_name]

            model = LGBMRegressor(**lgbm_params_optuna)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='rmse',
                      verbose=-1)
            y_preds = model.predict(X_val,
                                    num_iteration=best_iteration)

            RMSE = metrics.mean_squared_error(y_val, y_preds, squared=False)
            all_scores.append(RMSE)
    
    return np.mean(all_scores)