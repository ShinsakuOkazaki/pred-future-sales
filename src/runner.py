import numpy as np
import pandas as pd
from model import Model
from sklearn.metrics import mean_squared_error
from typing import Callable, List, Optional, Tuple, Union
import gc

from util import Logger, Util

logger = Logger()

class Runner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], features: List[str], params: dict):
        """コンストラクタ
 
        :param run_name: name of run
        :param model_cls: model class
        :param features: list of features
        :param params: hyper-parameter
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.n_fold = 7
        self.fold_index = [16, 19, 22, 25, 28, 31, 33]

    def train_fold(self, i_fold: Union[int, str], path: str) -> Tuple[
        Model, Optional[np.array], Optional[np.array]]:
        """cross-validation or hold-one by specifying a fold 

        :param ifold: number of fold
        :return tuple of instance of model, prediction, score
        """

        validation = i_fold != 'all'
        path = '../data' + path
        x_train, y_train = load_x_y_train(path)

        if validation:
            x_tr, y_tr = x_train[x_train.date_block_num < i_fold], y_train[y_train.date_block_num < i_fold]
            x_va, y_va = x_train[x_train.date_block_num == i_fold], y_train[y_train.date_block_num == i_fold]

            model = self.build_model(i_fold)
            model.train(x_tr, y_tr, x_va, y_va)

            pred_va = model.predict(x_va)
            score = mean_squared_error(y_va, pred_va)

            return model, pred_va, score
        
        else:
            model = self.build_model(i_fold)
            model.train(x_train, y_train)

            return Model, None, None
    
    def run_train_cv(self) -> None:
        """train and evaluate with cross-validation
        train and evaluate with each fold and save the model 
        """
        logger.info(f'{self.run_name} - start traing cv')

        scores = []
        preds = []
        data_block_nums = []
        for i_fold in range(self.n_fold):
            # train model
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, pred_va, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # save model
            model.save_model()

            # save result
            preds.append(pred_va)
            scores.append(score)
            date_block_nums.append([self.fold_index(i_fold) for in range(len(pred_va))])
        # concatenate all results
        preds = np.concatenate(preds, axis=0)
        date_block_nums = np.concatenate(data_block_nums, axis=0)
        preds = pd.DataFrame(data={'date_block_num': data_block_nums, 'item_cnt_month': preds})

        Util.dump(preds, f'../models/pred/{self.run_name}-train.pkl')

        logger.result_scores(self.run_name, scores)


    def run_train_all(self) -> None:
        """train with all data and save the model
        """

        logger.info(f'{self.run_name} - start training all')

        # train with all data
        i_fold = 'all'
        model, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self, path: str) -> None:
        """train all data and predict for test data
        run run_train_all ahead of this method
        pram: path: path to test data
        """

        logger.info(f'{self.run_name} - start prediction all')

        x_test = self.load_x_test(path)

        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load(model)
        pred = model.predict(x_test)

        # save result of prediction
        Util.dump(pred, f'../models/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def load_x_y_train(path: str) -> (pd.DataFrame, pd.Series):
        """load features of traning data

        :return: features of training data
        """
        path = '../data' + path
        train = pd.read_pickle(path)
        x_train = train.drop([self.features], axis=1, inplace=True)
        y_train = train.drop(['item_cnt_month'], axis=1, inplace=True)
        del train
        gc.collect();
        return x_train, y_train

    def load_x_test(path: str) -> pd.DataFrame:
        """load features of test data

        :return: features of test data
        """

        path = '../data' + path
        test = pd.read_pickle(path)
        x_test = test.drop([self.features], axis=1, inplace=True)
        del test
        gc.collect();
        return x_test

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """create model specifying a fold

        :param i_fold: number of fold
        :return: model instance
        """

        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params)

