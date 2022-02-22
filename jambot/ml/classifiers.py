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
import numpy as np
import pandas as pd
# from distributed import LocalCluster
# from lightgbm import DaskLGBMClassifier as LGBMClassifier
from lightgbm import LGBMClassifier, Sequence

from jambot import getlog
from jambot.common import DynConfig
from jambot.utils.mlflow import MlflowLoggable

if TYPE_CHECKING:
    from dask.dataframe.core import DataFrame as DDF

log = getlog(__name__)

warnings.filterwarnings('ignore', message='Parameter .* will be ignored')
warnings.filterwarnings('ignore', message='Will use it instead of argument')


class DaskSequence(Sequence):
    def __init__(self, ddf: 'DDF', batch_size: int = 10000):
        """
        Construct a sequence object from HDF5 with required interface.
        Parameters
        ----------
        ddf : dask.dataframe
        batch_size : int
            Size of a batch. When reading data to construct lightgbm Dataset, each read reads batch_size rows.
        """
        # We can also open HDF5 file once and get access to

        # reset index of dask df will be 0...n PER PARTITION
        self.ddf = ddf.reset_index(drop=True) \
            .pipe(monotonic_index) \
            .pipe(repartition, batch_size=batch_size)

        self._len = len(self.ddf)

        self.batch_size = batch_size
        self.active_block = pd.DataFrame()
        self.active_block_idx = -1

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        """Get values from ddf as computed np array vals
        - NOTE cannot call range index which overlaps block size eg 99_999:100_001

        Parameters
        ----------
        idx : Union[int, slice]
            eg 3 or 10004:10032

        Returns
        -------
        np.ndarray
            1d or 2d np array
        """

        # find block number that index intersects (int or slice)
        if not isinstance(idx, slice):
            i_block = idx // self.batch_size
            # offset = self.batch_size * i_block
            # idx = idx - offset
        else:
            i_block = idx.start // self.batch_size
            # offset = self.batch_size * i_block
            # idx = slice(idx.start - offset, idx.stop - offset)

        # load new block
        if not i_block == self.active_block_idx:
            self.active_block = self.ddf.partitions[i_block].compute()  # type: pd.DataFrame
            self.active_block_idx = i_block

        # lgbm dataset from samples needs values to be np.float64
        return self.active_block.loc[idx].values.astype(np.float64)

    def __len__(self):
        return self._len


class LGBMClsLog(LGBMClassifier, MlflowLoggable, DynConfig):
    """Wrapper to make LGBMClassifier easily loggable with mlflow"""

    log_keys = [
        'boosting_type',
        'num_leaves',
        'max_depth',
        'learning_rate',
        'n_estimators']

    # map sklearn: native api
    # https://stackoverflow.com/questions/47038276/lightgbm-sklearn-and-native-api-equivalence
    m_params = dict(
        num_leaves='num_leaves',
        n_estimators='num_boost_round',
        max_depth='max_depth',
        learning_rate='learning_rate',
        min_sum_hessian_in_leaf='min_child_weight',
        min_data_in_leaf='min_child_samples'
    )

    @property
    def log_items(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k in self.log_keys}

    def fit_iter(
            self,
            x_train: 'DDF',
            y_train: pd.Series,
            sample_weight: pd.Series = None,
            batch_size: int = 100000,
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
            # force_row_wise=True,
        )
        train_params.pop('silent')
        seq = DaskSequence(ddf=x_train, batch_size=batch_size)

        ds_params = dict(bin_construct_sample_cnt=200000, max_bin=255)  # defaults

        # save train booster to self
        self._Booster = lgb.train(
            params=train_params,
            # init_model=model,
            train_set=lgb.Dataset(
                data=[seq],
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
