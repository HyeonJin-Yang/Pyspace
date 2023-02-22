import warnings
import datetime
warnings.simplefilter('ignore')

from sklearn.metrics import mean_squared_error

from utils import *

train = pd.read_csv('./input/PG_S03E06/train_df.csv')

xgb_path = './output/run_XGBR_' + '230214134832'
lgbm_path = './output/run_LGBR_' + '230214135321'

xgb_oof = pd.read_csv(xgb_path + '/oof.csv')
lgbm_oof = pd.read_csv(lgbm_path + '/oof.csv')
xgb_preds = pd.read_csv(xgb_path + '/submission.csv')
lgb_preds = pd.read_csv(lgbm_path + '/submission.csv')

ensemble_pred = train[[idx_name]]
ensemble_pred[target_name] = 0
ensemble_scores = []
for w in np.arange(0.0, 1.01, 0.01):
    ensemble_pred[target_name] = w * xgb_oof[target_name] + (1-w) * lgbm_oof[target_name]
    ensemble_rmse = mean_squared_error(train[target_name], ensemble_pred[target_name], squared=False)
    ensemble_scores.append(ensemble_rmse)
best_score = np.min(ensemble_scores)
best_weight = np.argmin(ensemble_scores) / 100

sub = pd.read_csv('./input/PG_S03E06/sample_submission.csv')
sub[target_name] = best_weight * xgb_preds[target_name] + (1-best_weight) * lgb_preds[target_name]

output_path = './output/Ensemble/' + datetime.datetime.now().strftime('%y%m%d%H%M%S')
if not os.path.exists(output_path):
    os.mkdir(output_path)
sub.to_csv(output_path + '/submission_ensemble.csv', index=False)