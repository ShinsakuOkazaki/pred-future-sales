import numpy as pn
import pandas as pd

from model_nn import ModelNN
from runner import Runner
from util import Submission

if __name__ == '__main__':

    params_nn = {
        'input_dropout': 0.0, 
        'hidden_layers': 3, 
        'hidden_units': 96,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.2, 
        'batch_norm': 'before_act', 
        'optimizer': {'type': 'adam', 'lr': 0.001}, 
        'batch_size': 64, 
        }


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

    runner = Runner('nn1', ModelNN, features, params_nn, train_path, test_path)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('xgb1', 0)
    Submission.create_submission('xgb1', 1)
    Submission.create_submission('xgb1', 2)
    Submission.create_submission('xgb1', 3)
    Submission.create_submission('xgb1', 4)
    Submission.create_submission('xgb1', 5)
    Submission.create_submission('xgb1', 6)

    # runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb, train_path, test_path)
    # runner.run_train_all()
    # runner.run_predict_all()
    # Submission.create_submission('xgb1-train-all')

    
