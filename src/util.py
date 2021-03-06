import datetime
import logging
import os

import numpy as np
import pandas as pd
import joblib

class Util:

    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)        
        joblib.dump(value, path, compress=True)
    @classmethod
    def load(cls, path):
        return joblib.load(path)


class Logger:

    def __init__(self):
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler('../models/general.log')
        file_result_handler = logging.FileHandler('../models/result.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
       dic = dict() 
       dic['name'] = run_name
       dic['score'] = np.mean(scores)
       for i, score in enumerate(scores):
           dic[f'score{i}'] = score
       self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Submission:

    @classmethod
    def create_submission(cls, run_name, i_fold=None):
        submission = pd.read_csv('../data/raw/sample_submission.csv')
        if i_fold:
            run_fold_name = f'{run_name}-{i_fold}'
        else:
            run_fold_name = f'{run_name}-test'
       
        pred = Util.load(f'../models/pred/{run_fold_name}.pkl')
        submission['item_cnt_month'] = pred
        submission.to_csv(f'../submission/{run_fold_name}.csv', index=False)

    
