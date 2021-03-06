import numpy as np
import pandas as pd
import gc
import xgboost as xgb
from sklearn.metrics import mean_squared_error
working_dir = '~/Project/pred-future-sales'

params = {
        'booster': 'gbtree', 
        'objective': 'reg:linear', 
        'eta': 0.1, 
        'gamma': 0.0, 
        'alpha': 0.0, 
        'lambda': 1.0, 
        'min_child_weight': 1, 
        'max_depth': 5, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'random_state': 71, 
        }
num_round = 50

if __name__ == '__main__':
    train_test = pd.read_pickle(working_dir + '/data/interim/train_test.pkl')
    test = pd.read_csv(working_dir + '/data/raw/test.csv')
    x_train = train_test[train_test.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    y_train = train_test[train_test.date_block_num < 33]['item_cnt_month']
    x_valid = train_test[train_test.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    y_valid = train_test[train_test.date_block_num == 33]['item_cnt_month']
    x_test = train_test[train_test.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    
    del train_test

    gc.collect();

    dtrain = xgb.DMatrix(x_train, y_train)
    dvalid = xgb.DMatrix(x_valid, y_valid)
    dtest = xgb.DMatrix(x_test)

    watchlist = [(dtrain, 'train', (dvalid, 'eval'))]
    model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=50)
    va_pred = model.predict(dvalid)
    score = mean_squared_error(y_valid, va_pred)
    print('rsme: {}'.format(score))

    pred = model.predict(dtest)

    submission = pd.DataFrame({
        'ID' : test.index, 
        'item_cnt_month': pred
        })

    submission.to_csv(working_dir + '/submission/init_xgb_submission.csv', index=False)



