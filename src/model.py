import pandas as pd
import numpy as np
from abs import ABCMeta, abstractmethod
from typing import Optional

class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """constructor

        :param run_fold_name: combination of run_name and fold_number
        :param param: hyper-parameter
        """

        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None
    
    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series, 
              va_x: Optional[pd.DataFrame] = None, 
              va_y: Optional[pd.DataFrame] = None) -> None:
        """trains model and save trained model

        :param tr_x: features of training data 
        :param tr_y: target of training data
        :param va_x: features of validation data
        :param va_y: target of validation data
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """predict with trained model

        :param te_x: features from validation or test data
        :return: prediction
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """save model"""
        pass

    @abstractmethod
    def loadd_model(self) -> None:
        """load model"""
        pass
