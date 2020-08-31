import os
import numpy as np
import pandas as pd
import xgboost as xgb

from model import Model
from util import Util

class ModelXGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # set data
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            dvalid = xgb.Dmatrix(va_x, va_y)
        
        # configurate parameter
        params = dict(self.params)
        num_round = params.pop('num_round')

        # training
        if validation: 
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
                                   early_stopping_rounds=early_stopping_rounds)
        
        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(params, dtrain, num_round, eval=watchlist)

        
    def predict(self, te_x):
        dtest = xgb.Dmatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

    def save_model(self):
        model_path = os.path.join('../models/model', f'{self.run_fold_name}.model') 
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../models/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
