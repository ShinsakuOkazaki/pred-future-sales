import numpy as pn
import pandas as import pd

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
        'colsample_bytree': 0.8, 
        'random_state': 71, 
        }

    params_xgb['num_round'] = 50


    features = ['date_block_num', 
                 'shop_id', 
                 'item_id', 
                 'item_cnt_month', 
                 'city_code', 
                 'item_category_id', 
                 'type_code', 
                 'subtype_code',
                 'item_cnt_month_lag_1'
                 'item_cnt_month_lag_2',
                 'item_cnt_month_lag_3', 
                 'item_cnt_month_lag_6'
                 'item_cnt_month_lag_12'
                 'item_cnt_month_item_avg_lag_1'
                 'item_cnt_month_item_avg_lag_2'
                 'item_cnt_month_item_avg_lag_3'
                 'item_cnt_month_item_avg_lag_6'
                 'item_cnt_month_item_avg_lag_12'
                 'item_cnt_month_shop_avg_lag_1'
                 'item_cnt_month_shop_avg_lag_2'
                 'item_cnt_month_shop_avg_lag_3'
                 'item_cnt_month_shop_avg_lag_6'
                 'item_cnt_month_shop_avg_lag_12'
                 'item_cnt_month_category_avg_lag_1'
                 'item_cnt_month_category_avg_lag_2'
                 'item_cnt_month_category_avg_lag_3'
                 'item_cnt_month_category_avg_lag_6'
                 'item_cnt_month_category_avg_lag_12'
                 'item_cnt_month_type_avg_lag_1'
                 'item_cnt_month_subtype_avg_lag_1'
                 'item_cnt_month_city_avg_lag_1'
                 'item_cnt_month_shop_type_avg_lag_1'
                 'item_cnt_month_shop_category_avg_lag_1'                
                 'item_cnt_month_shop_subtype_avg_lag_1'
                 'item_cnt_month_item_city_avg_lag_1'
                 'delta_item_price_lag'
                ]

    runner = Runner('xgb1', ModelXGB, features, params_xgb)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('xgb1')

    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb)
    runner.run_train_all()
    runner.run_test_all()
    Submission.create_submission('xgb1-train-all')

    