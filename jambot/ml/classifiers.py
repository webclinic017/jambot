from typing import *

from lightgbm import LGBMClassifier

from jambot import config as cf
from jambot.utils.mlflow import MlflowLoggable


class LGBMClsLog(LGBMClassifier, MlflowLoggable):
    """Wrapper to make LGBMClassifier easily loggable with mlflow"""

    log_keys = [
        'boosting_type',
        'num_leaves',
        'max_depth',
        'learning_rate',
        'n_estimators']

    @property
    def log_items(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k in self.log_keys}

    @classmethod
    def from_config(cls, symbol: str = cf.SYMBOL, **kw) -> 'LGBMClsLog':
        """Instantiate from dynamic config file"""
        kw = cf.dynamic_cfg(symbol, keys=cls.log_keys) | kw
        return cls(**kw)
