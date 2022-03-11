"""
LGBM dask issues
https://github.com/microsoft/LightGBM/issues/4229
https://github.com/microsoft/LightGBM/issues/4625

info
https://odsc.medium.com/scaling-lightgbm-with-dask-265c1a0ae79f
https://towardsdatascience.com/how-to-learn-from-bigdata-files-on-low-memory-incremental-learning-d377282d38ff

from distributed import Client, LocalCluster
n_workers = 1
# cluster = LocalCluster(n_workers=n_workers)
client = Client(processes=False)
# client.wait_for_workers(n_workers)
model.set_params(client=client)
"""


import warnings
from typing import *

import lightgbm as lgb
import pandas as pd
from lightgbm.sklearn import LGBMClassifier

from jambot import getlog
from jambot.common import DynConfig
from jambot.utils.mlflow import MlflowLoggable

if TYPE_CHECKING:
    from dask.dataframe.core import DataFrame as DDF

    from jambot.ml.dataloading import DiskDataFrame

log = getlog(__name__)

warnings.filterwarnings('ignore', message='Parameter .* will be ignored')
warnings.filterwarnings('ignore', message='Will use it instead of argument')
warnings.filterwarnings('ignore', message='No further splits with positive gain, best gain')


class LGBMClsLog(LGBMClassifier, MlflowLoggable, DynConfig):
    """Wrapper to make LGBMClassifier easily loggable with mlflow"""
    log_keys = [
        'boosting_type',
        'num_leaves',
        'max_depth',
        'learning_rate',
        'n_estimators',
        'random_state']

    # map sklearn: native api
    # https://stackoverflow.com/questions/47038276/lightgbm-sklearn-and-native-api-equivalence
    m_params = dict(
        num_leaves='num_leaves',
        n_estimators='num_boost_round',  # number of individual trees
        max_depth='max_depth',
        learning_rate='learning_rate',
        min_sum_hessian_in_leaf='min_child_weight',
        min_data_in_leaf='min_child_samples')

    @property
    def log_items(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k in self.log_keys and not k == 'random_state'}

    def fit_iter(
            self,
            ddf_x_train: 'DiskDataFrame',
            y_train: pd.Series,
            sample_weight: pd.Series = None,
            *args, **kw):
        """Use native lgb.train() interface instead of sklearn (for continued learning)
        - NOTE not used

        Parameters
        ----------
        ddf : DDF
            dask dataframe
        sample_weight : pd.Series, optional
            _description_, by default None

        Returns
        -------
        LGBMClsLog
            self with fit _Booster

        Params Example
        --------------
        >>> self.get_params()
        >>> {
            'boosting_type': 'dart',
            'class_weight': None,
            'colsample_bytree': 1.0,
            'importance_type': 'split',
            'learning_rate': 0.1,
            'max_depth': 29,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'n_estimators': 83,
            'n_jobs': -1,
            'num_leaves': 47,
            'objective': None,
            'random_state': None,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'silent': 'warn',
            'subsample': 1.0,
            'subsample_for_bin': 200000,
            'subsample_freq': 0}
        """
        train_params = self.get_params() | dict(
            objective='binary',
            # keep_training_booster=True,
            # verbosity=0,
            force_row_wise=True,
            # force_col_wise=True,  # uses les memory?
        )
        train_params.pop('silent')
        train_params.pop('subsample_for_bin')
        train_params.pop('importance_type')

        ds_params = dict(bin_construct_sample_cnt=200000, max_bin=255)  # defaults

        ddf_x_train.init_lgbm_seqs()

        # save train booster to self
        self._Booster = lgb.train(
            params=train_params,
            train_set=lgb.Dataset(
                data=ddf_x_train.seqs,
                label=y_train,
                weight=sample_weight,
                params=ds_params),
        )


def monotonic_index(ddf: 'DDF') -> 'DDF':
    """Create monotonic increasing index on dask df
    - annoying this is so complicated

    Parameters
    ----------
    ddf : DDF

    Returns
    -------
    DDF
    """
    import dask as da
    cumlens = ddf.map_partitions(len).compute().cumsum()

    new_partitions = [ddf.partitions[0]]
    for npart, partition in enumerate(ddf.partitions[1:].partitions):
        partition.index = partition.index + cumlens[npart]
        new_partitions.append(partition)

    ddf = da.dataframe.concat(new_partitions)
    return ddf.set_index(ddf.index, sorted=True)


def repartition(ddf: 'DDF', batch_size: int) -> 'DDF':
    """Convenience func to repartion ddf with fixed batch sizes

    Parameters
    ----------
    ddf : DDF
    batch_size : int
        eg 50_000

    Returns
    -------
    DDF
    """
    num_partitions = len(ddf) // batch_size + 2
    divs = [n * batch_size for n in range(num_partitions)]
    return ddf.repartition(divisions=divs, force=True)
