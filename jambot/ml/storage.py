from datetime import datetime as dt
from datetime import timedelta as delta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from jambot import SYMBOL
from jambot import config as cf
from jambot import functions as f
from jambot import getlog
from jambot import sklearn_utils as sk
from jambot.common import DictRepr
from jambot.database import db
from jambot.ml import models as md
from jambot.signals import WeightedPercent
from jambot.utils.storage import BlobStorage

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
            test: bool = False):
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
        container = 'jambot-app'

        if test:
            container = f'{container}-test'

        # init BlobStorage to mirror local data dir to azure blob storage
        bs = BlobStorage(container=container)

        dt_format = '%Y-%m-%d %H'
        dt_format_path = '%Y-%m-%d-%H'
        batch_size_cdls = batch_size

        if interval == 15:
            batch_size_cdls = batch_size_cdls * 4
            dt_format_path = f'{dt_format_path}-%M'
            dt_format = f'{dt_format}:%M'

        p_model = cf.p_data / 'models'
        f.check_path(p_model)

        if d_lower is None:
            d_lower = cf.config['d_lower']

        f.set_self(vars())

    def clean(self) -> None:
        """Clean all saved models in models dir"""
        for p in self.p_model.glob('*'):
            p.unlink()

    def fit_save_models(
            self,
            df: pd.DataFrame = None,
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
            df=db.get_df(symbol=SYMBOL, startdate=self.d_lower, interval=interval)
            .pipe(md.add_signals, name=name, drop_ohlc=False)
            .iloc[:-1 * n_periods, :],
            name=name,
            estimator=estimator)

    def fit_save(self, df: pd.DataFrame, name: str, estimator: BaseEstimator) -> None:
        """fit model and save for n_models

        - to be run every x hours
        - all models start fit from same d_lower
        - new versions created for each new saved ver

        Parameters
        ----------
        df : pd.DataFrame
            df with signals/target initialized
        estimator : BaseEstimator
            estimator with fit/predict methods
        """
        log.info('running fit_save')
        cfg = md.model_cfg(name)

        # max date where hour is greater or equal to 18:00
        # set back @cut hrs due to losing @n_periods for training preds
        # model is trained AT 1800, but only UP TO 1500
        cut_hrs = {1: 10, 15: 3}.get(self.interval)
        reset_hour_offset = self.reset_hour - cut_hrs  # 15

        d_upper = f.date_to_dt(
            df.query('timestamp.dt.hour >= @reset_hour_offset').index.max().date()) \
            + delta(hours=reset_hour_offset)

        # print('d_upper: ', d_upper)

        # get weights for fit params
        weights = WeightedPercent(cfg['n_periods_weighted']).get_weight(df).loc[:d_upper]

        index = df.loc[:d_upper].index
        df = df \
            .pipe(f.safe_drop, cols=cf.config['drop_cols']) \
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
                **sk.weighted_fit(name=None, weights=weights.iloc[:upper])
                # **sk.weighted_fit(name=None, n=len(df) + cut_rows)
            )

            # save - add back cut hrs so always consistent
            # d = date model was trained
            d = index[cut_rows - 1] + delta(hours=cut_hrs)  # + delta(days=1)
            fname = f'{name}_{d:{self.dt_format_path}}'
            f.save_pickle(estimator, p=self.p_model, name=fname)
            log.info(f'saved model: {fname}')

        # mirror saved models to azure blob
        self.bs.upload_dir(p=self.p_model, mirror=True)

    def df_pred_from_models(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Load all models, make iterative predictions
        - probas added in a bit of a unique way here, one model slice at a time instead of all at once

        Parameters
        ----------
        df : pd.DataFrame
            truncated df (~400 rows?) with signals added and OHLC cols dropped,/*----- for live trading
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
            estimator = f.load_pickle(p)

            if not i == len(p_models) - 1:
                max_date = d + delta(hours=self.batch_size) - f.get_offset(self.interval)
            else:
                max_date = df.index[-1]

            # split into 24 hr slices
            x, _ = sk.split(df.loc[d: max_date], target=target)

            # add y_pred and proba_ for slice of df
            df_pred = sk.df_proba(
                x=x.pipe(f.safe_drop, cols=cf.config['drop_cols']),
                model=estimator)

            idx = df_pred.index
            fmt = self.dt_format
            log.info(f'Adding preds: {len(df_pred):02}, {idx.min():{fmt}}, {idx.max():{fmt}}')

            pred_dfs.append(df_pred)

        return df.pipe(f.left_merge, pd.concat(pred_dfs)) \
            .pipe(md.add_proba_trade_signal)

    def to_dict(self):
        return ('d_lower', 'n_models', 'batch_size', 'batch_size_cdls')


if __name__ == '__main__':
    ModelStorageManager().fit_save_models()
