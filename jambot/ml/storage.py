from datetime import datetime as dt
from datetime import timedelta as delta
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from jambot import SYMBOL
from jambot import config as cf
from jambot import functions as f
from jambot import getlog
from jambot import sklearn_utils as sk
from jambot.common import DictRepr
from jambot.ml import models as md
from jambot.tables import Tickers
from jambot.weights import WeightsManager
from jgutils import fileops as jfl
from jgutils import pandas_utils as pu
from jgutils.azureblob import BlobStorage

if TYPE_CHECKING:
    from pathlib import Path

    from lightgbm import LGBMClassifier

    from jambot.livetrading import ExchangeManager

log = getlog(__name__)


class ModelStorageManager(DictRepr):
    """Class to manage saving/training/loading ml models"""

    def __init__(
            self,
            batch_size: int = 24,
            n_models: int = 3,
            d_lower: dt = None,
            interval: int = 15,
            reset_hour: int = 18,
            test: bool = False,
            *args,
            **kwargs):
        """
        Parameters
        ----------
        batch_size : int, optional
            number of hours to retrain after, default 24
        n_models : int, optional
            number of models to train at once, default 3
        d_lower : dt, optional
            trim df to lower date, by default None
        interval : int, optional
            interval 1hr/15min
        reset_hour : int, optional
            filter training df to constant hour per day for consistency (11 pst = 18 utc)
        """
        self.batch_size = batch_size
        self.n_models = n_models
        self.d_lower = d_lower
        self.interval = interval
        self.reset_hour = reset_hour
        self.test = test
        self.container = 'jambot-app'
        self.saved_models = []  # type: List[Path]

        if self.test:
            self.container = f'{self.container}-test'

        # init BlobStorage to mirror local data dir to azure blob storage
        self.bs = BlobStorage(container=self.container)

        self.dt_format = '%Y-%m-%d %H'
        self.dt_format_path = '%Y-%m-%d-%H'
        self.batch_size_cdls = self.batch_size

        if interval == 15:
            self.batch_size_cdls = self.batch_size_cdls * 4
            self.dt_format_path = f'{self.dt_format_path}-%M'
            self.dt_format = f'{self.dt_format}:%M'

        self.p_model = jfl.check_path(cf.p_data / 'models')

        if self.d_lower is None:
            self.d_lower = cf.D_LOWER

    def clean(self) -> None:
        """Clean all saved models in models dir"""
        jfl.clean_dir(self.p_model)

    def fit_save_models(
            self,
            em: 'ExchangeManager',
            interval: int = 15) -> None:
        """Main control function for retraining new models each day

        Parameters
        ----------
        df : pd.DataFrame, optional
            df with OHLC from db, by default None
        interval : int, optional
            default 15
        """
        log.info('running fit_save_models')
        name = 'lgbm'
        cfg = md.model_cfg(name)
        n_periods = cfg['target_kw']['n_periods']

        self.clean()

        estimator = md.make_model(name)

        self.fit_save(
            df=Tickers().get_df(
                symbol=SYMBOL,
                startdate=self.d_lower,
                interval=interval,
                funding=True,
                funding_exch=em.default('bitmex'))
            .pipe(md.add_signals, name=name, drop_ohlc=False, use_important=True)
            .iloc[:-1 * n_periods, :],
            name=name,
            estimator=estimator)

    def fit_save(self, df: pd.DataFrame, name: str, estimator: 'LGBMClassifier') -> None:
        """fit model and save for n_models

        - to be run every x hours
        - all models start fit from same d_lower
        - new versions created for each new saved ver

        Parameters
        ----------
        df : pd.DataFrame
            df with signals/target initialized
        estimator : LGBMClassifier
            estimator with fit/predict methods
        """
        cfg = md.model_cfg(name)

        # max date where hour is greater or equal to 18:00
        # set back @cut hrs due to losing @n_periods for training preds
        # model is trained AT 1800, but only UP TO 1500
        cut_hrs = {1: 10, 15: 3}[self.interval]
        reset_hour_offset = self.reset_hour - cut_hrs  # 15

        d_upper = f.date_to_dt(
            df.query('timestamp.dt.hour >= @reset_hour_offset').index.max().date()) \
            + delta(hours=reset_hour_offset)

        # get weights for fit params

        weights = WeightsManager.from_config(df).weights.loc[:d_upper]

        index = df.loc[:d_upper].index
        df = df \
            .pipe(pu.safe_drop, cols=cf.DROP_COLS) \
            .loc[:d_upper] \
            .to_numpy(np.float32)

        # trim df to older dates by cutting off progressively larger slices
        for i in range(self.n_models):

            cut_rows = -i * self.batch_size_cdls
            upper = cut_rows or None

            # fit - using weighted currently
            estimator.fit(
                df[:upper, :-1],
                df[:upper, -1],
                sample_weight=weights.iloc[:upper]
            )

            # save - add back cut hrs so always consistent
            # d = date model was trained
            d = index[cut_rows - 1] + delta(hours=cut_hrs)  # + delta(days=1)
            fname = f'{name}_{d:{self.dt_format_path}}'
            p_save = jfl.save_pickle(estimator, p=self.p_model, name=fname)
            self.saved_models.append(p_save)
            log.info(f'saved model: {fname}')

        # mirror saved models to azure blob
        self.bs.upload_dir(p=self.p_model, mirror=True)

    def df_pred_from_models(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Load all models, make iterative predictions
        - probas added in a bit of a unique way here, one model slice at a time instead of all at once

        Parameters
        ----------
        df : pd.DataFrame
            truncated df (~400 rows?) with signals added and OHLC cols dropped, for live trading
        name : str
            model name

        Returns
        -------
        pd.DataFrame
            df with all predictions added
        """
        self.bs.download_dir(p=self.p_model, mirror=True)

        cfg = md.model_cfg(name)
        target = cfg['target']
        pred_dfs = []

        p_models = sorted(self.p_model.glob(f'*{name}*'))

        if len(p_models) == 0:
            raise RuntimeError(f'No saved models found at: {self.p_model}')

        for i, p in enumerate(p_models):

            # get model train date from filepath
            d = dt.strptime(p.stem.split('_')[-1], self.dt_format_path)

            # load estimator
            estimator = jfl.load_pickle(p)  # type: LGBMClassifier

            if not i == len(p_models) - 1:
                max_date = d + delta(hours=self.batch_size) - f.inter_offset(self.interval)
            else:
                max_date = df.index[-1]

            # split into 24 hr slices
            x, _ = sk.split(df.loc[d: max_date], target=target)

            # btwn 18:00 to 18:15 last df won't have any rows
            # eg new model has been trained, but next candle not imported yet
            if len(x) > 0:
                # add y_pred and proba_ for slice of df
                df_pred = sk.df_proba(
                    x=x.pipe(pu.safe_drop, cols=cf.DROP_COLS),
                    model=estimator)

                idx = df_pred.index
                fmt = self.dt_format
                log.info(f'Adding preds: {len(df_pred):02}, {idx.min():{fmt}}, {idx.max():{fmt}}')

                pred_dfs.append(df_pred)

        return df.pipe(pu.left_merge, pd.concat(pred_dfs)) \
            .pipe(md.add_proba_trade_signal)

    def to_dict(self):
        return ('d_lower', 'n_models', 'batch_size', 'batch_size_cdls')
