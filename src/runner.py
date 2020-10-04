import numpy as np
import pandas as pd
from model import Model
from sklearn.metrics import mean_squared_error
from typing import Callable, List, Optional, Tuple, Union
import gc

from util import Logger, Util

logger = Logger()

class Runner:

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], features: List[str], params: dict, train_path: str, test_path):
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
        self.date_block_nums = [16, 19, 22, 25, 28, 31, 33]
        self.train_path = train_path
        self.test_path = test_path

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[np.array]]:
        """cross-validation or hold-one by specifying a fold 

        :param ifold: number of fold
        :return tuple of instance of model, prediction, score
        """
        validation = i_fold != 'all'
        x_train, y_train = self.load_x_y_train()
        
        if validation:
            date_block_num = self.date_block_nums[i_fold]
            tr_idx, va_idx = self.load_index_fold(x_train, date_block_num)
            x_tr, y_tr = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
            x_va, y_va = x_train.iloc[va_idx], y_train.iloc[va_idx]

            model = self.build_model(i_fold)
            model.train(x_tr, y_tr, x_va, y_va)

            pred_va = model.predict(x_va)
            score = mean_squared_error(y_va, pred_va)

            return model, va_idx, pred_va, score
        
        else:
            model = self.build_model(i_fold)
            model.train(x_train, y_train)

            return model, None, None, None
    
    def run_train_cv(self) -> None:
        """train and evaluate with cross-validation
        train and evaluate with each fold and save the model 
        """
        logger.info('{} - start traing cv'.format(self.run_name))

        scores = []
        preds = []
        va_idxes = []
        for i_fold in range(self.n_fold):
            # train model
            logger.info('{} fold {} - start training'.format(self.run_name, i_fold))
            model, va_idx, pred_va, score = self.train_fold(i_fold)
            logger.info('{} fold {} - end training - score {}'.format(self.run_name, i_fold, score))

            # save model
            model.save_model()

            # save result
            va_idxes.append(va_idx)
            preds.append(pred_va)
            scores.append(score)
        # concatenate all results
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info('{} - end training cv - score {}'.format(self.run_name, np.mean(scores)))
        Util.dump(preds, f'../models/pred/{self.run_name}-train.pkl')

        logger.result_scores(self.run_name, scores)

    def run_train_cv_each(self, i_fold: Union[int, str]) -> None:

        logger.info(f'{self.run_name} - start traing cv')
        # train model
        logger.info(f'{self.run_name} fold {i_fold} - start training')
        model, va_idx, pred_va, score = self.train_fold(i_fold)
        logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')
        # save model
        model.save_model()
        
        Util.dump(pred_va, f'../models/pred/{self.run_name}-train.pkl')

        logger.result_scores(self.run_name, score)

    def run_train_all(self) -> None:
        """train with all data and save the model
        """

        logger.info(f'{self.run_name} - start training all')

        # train with all data
        i_fold = 'all'
        model, _, _, _  = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_cv(self) -> None:

        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.load_x_test()

        preds = []
        for i_fold in range(self.n_fold):

            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            Util.dump(pred, f'../models/pred/{self.run_name}-{i_fold}.pkl')
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.dump(pred_avg, f'../models/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

    def run_predict_all(self) -> None:
        """train all data and predict for test data
        run run_train_all ahead of this method
        pram: path: path to test data
        """

        logger.info(f'{self.run_name} - start prediction all')

        x_test = self.load_x_test()

        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(x_test)

        # save result of prediction
        Util.dump(pred, f'../models/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction all')



    def load_x_y_train(self) -> (pd.DataFrame, pd.Series):
        """load features of traning data

        :return: features of training data
        """
        path = '../data' + self.train_path 
        train = pd.read_pickle(path)
        x_train = train[self.features]
        y_train = train['item_cnt_month']
        
        return x_train, y_train
    
    def load_x_test(self) -> pd.DataFrame:
        """load features of test data

        :return: features of test data
        """

        path = '../data' + self.test_path
        test = pd.read_pickle(path)
        x_test = test[self.features]
        return x_test

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """create model specifying a fold

        :param i_fold: number of fold
        :return: model instance
        """

        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params)

    def load_index_fold(self, x_train: pd.DataFrame, date_block_num: int) -> (np.array, np.array):
        
        tr_index = x_train.index[x_train.date_block_num < date_block_num].to_numpy()
        va_index = x_train.index[x_train.date_block_num == date_block_num].to_numpy()

        return tr_index, va_index

