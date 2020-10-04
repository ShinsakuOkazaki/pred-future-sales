import numpy as pn
import pandas as pd

from model_xgb import ModelXGB
from runner import Runner
from util import Submission

if __name__ == '__main__':

    params_xgb = {
        'booster': 'gbtree', 
        'objective': 'reg:linear', 
        'eta': 0.1, 
        'gamma': 0.0, 
        'alpha': 0.0, 
        'lambda': 1.0, 
        'min_child_weight': 1, 
        'max_depth': 5, 
        'subsample': 0.8,
        'early_stopping_rounds': 50,
        'colsample_bytree': 0.8, 
        'random_state': 71, 
        }

    params_xgb['num_round'] = 1000


    features = ['date_block_num', 
                 'shop_id', 
                 'item_id', 
                 'city_code', 
                 'item_category_id', 
                 'type_code', 
                 'subtype_code',
                 'item_cnt_month_lag_1',
                 'item_cnt_month_lag_2',
                 'item_cnt_month_lag_3', 
                 'item_cnt_month_lag_6', 
                 'item_cnt_month_lag_12', 
                 'item_cnt_month_item_avg_lag_1', 
                 'item_cnt_month_item_avg_lag_2', 
                 'item_cnt_month_item_avg_lag_3', 
                 'item_cnt_month_item_avg_lag_6', 
                 'item_cnt_month_item_avg_lag_12', 
                 'item_cnt_month_shop_avg_lag_1', 
                 'item_cnt_month_shop_avg_lag_2', 
                 'item_cnt_month_shop_avg_lag_3', 
                 'item_cnt_month_shop_avg_lag_6', 
                 'item_cnt_month_shop_avg_lag_12', 
                 'item_cnt_month_category_avg_lag_1', 
                 'item_cnt_month_category_avg_lag_2', 
                 'item_cnt_month_category_avg_lag_3', 
                 'item_cnt_month_category_avg_lag_6', 
                 'item_cnt_month_category_avg_lag_12', 
                 'item_cnt_month_type_avg_lag_1', 
                 'item_cnt_month_subtype_avg_lag_1', 
                 'item_cnt_month_city_avg_lag_1', 
                 'item_cnt_month_shop_type_avg_lag_1', 
                 'item_cnt_month_shop_category_avg_lag_1', 
                 'item_cnt_month_shop_subtype_avg_lag_1', 
                 'item_cnt_month_item_city_avg_lag_1', 
                 'delta_item_price_lag', 
                 ]
    train_path = '/interim/train.pkl'
    test_path = '/interim/test.pkl'

    
    runner = Runner('xgb2-train-all', ModelXGB, features, params_xgb, train_path, test_path)
    runner.run_train_all()
    runner.run_predict_all()
    Submission.create_submission('xgb2-train-all')

    
