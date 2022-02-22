import math
from datetime import datetime as dt
from datetime import timedelta as delta
from typing import *

import pandas as pd

from jambot import SYMBOL
from jambot import config as cf
from jambot import functions as f
from jambot import getlog
from jambot import sklearn_utils as sk
from jambot.common import DictRepr
from jambot.data import DataManager
from jambot.ml import models as md
from jambot.tradesys.symbols import Symbol
from jambot.weights import WeightsManager
from jgutils import fileops as jfl
from jgutils import pandas_utils as pu
from jgutils.azureblob import BlobStorage

if TYPE_CHECKING:
    from pathlib import Path

    from jambot.livetrading import ExchangeManager
    from jambot.ml.classifiers import LGBMClsLog

log = getlog(__name__)
NAME = 'lgbm'  # default model name


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
        self.dm = DataManager()

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

    def local_models(self, symbol: str = None) -> List['Path']:
        """Get sorted list of all local models

        Parameters
        ----------
        symbol : str, optional
            filter local models by symbol name
        """
        symbol = symbol or ''
        return sorted(self.p_model.glob(f'*{symbol.lower()}*'))

    def local_model_names(self, symbol: str = None) -> List[str]:
        """Get list of local model names to compare with azure storage
        - only used for testing

        Parameters
        ----------
        symbol : str, optional
            default 'XBTUSD'
        """
        return [p.name for p in self.local_models(symbol=symbol)]

    def get_oldest_model(self, symbol: str = SYMBOL) -> 'Path':
        """Get oldest model path

        Parameters
        ----------
        symbol : str, optional
            default 'XBTUSD'
        """
        return self.local_models(symbol)[0]

    def n_models_local(self, symbol: str = SYMBOL) -> int:
        """Count of local model paths"""
        return len(self.local_models(symbol))

    def parse_date(self, p: 'Path') -> dt:
        """Get date from model path"""
        return dt.strptime(p.stem.split('_')[-1], self.dt_format_path)

    def d_latest_model(self, symbol: str = SYMBOL) -> dt:
        """Get date of latest model"""
        p_models = self.local_models(symbol)
        if p_models:
            return self.parse_date(p=p_models[-1])
        else:
            # no saved models exist, return low date
            return dt(2000, 1, 1)

    def clean(self, last_only: bool = True, symbol: str = SYMBOL) -> None:
        """Clean all or oldest saved models in models dir

        Parameters
        ----------
        last_only : bool, optional
            only delete oldest model path, default True
        symbol : str, optional
            default 'XBTUSD'
        """
        if last_only:
            p = self.get_oldest_model(symbol=symbol)
            p.unlink()
        else:
            for p in self.local_models(symbol=symbol):
                p.unlink()

    def _get_model_fname(self, symbol: str, d: dt) -> str:
        """Create model path name string

        Parameters
        ----------
        symbol : str
        d : dt

        Returns
        -------
        str
            eg 'xbtusd_2022-01-31-18-00'
        """
        return f'{symbol.lower()}_{d:{self.dt_format_path}}'

    def get_model_path(self, symbol: str, d: dt) -> 'Path':
        return self.p_model / '{}.pkl'.format(self._get_model_fname(symbol=symbol, d=d))

    def save_model(self, model: 'LGBMClsLog', symbol: str, d: dt) -> 'Path':
        """Save model to pickle

        Parameters
        ----------
        model : LGBMClsLog
        symbol : str
        d : dt

        Returns
        -------
        Path
            path of saved model pickle
        """
        fname = self._get_model_fname(symbol=symbol, d=d)
        p_save = jfl.save_pickle(model, p=self.p_model, name=fname)
        log.info(f'saved model: {fname}')
        return p_save

    def load_model(self, p: 'Path' = None, symbol: str = None, d: dt = None) -> 'LGBMClsLog':
        """Load model by:
            1. path
            2. symbol/date

        Parameters
        ----------
        p : Path, optional
        symbol : str, optional
        d : dt, optional

        Returns
        -------
        LGBMClsLog
            trained model
        """
        if p is None:
            if symbol is None or d is None:
                raise RuntimeError('Need path or symbol/date to load model!')

            p = self.get_model_path(symbol=symbol, d=d)

        return jfl.load_pickle(p)

    def fit_save_models(
            self,
            em: 'ExchangeManager',
            # df: pd.DataFrame,
            symbol: Union[Symbol, str],
            name: str = NAME,
            interval: int = 15,
            overwrite_all: bool = False) -> None:
        """Retrain single new, or overwrite all models
        - run live every n hours (24)
        - models trained AT 18:00, but only UP TO eg 10:00

        Parameters
        ----------
        em : ExchangeManager
        df : pd.DataFrame, optional
            df with OHLC from db, default None
        symbol : Union[Symbol, str]
            eg Symbol('XBTUSD', 'bitmex') or 'MULTI_ALTS'
        name : str, optional
            model name, default 'lgbm'
        interval : int, optional
            default 15
        overwrite_all : bool, optional
            If True delete all existing models and retrain,
            else retrain newest only, default False
        interval : int, optional
            default 15
        """
        self.bs.download_dir(p=self.p_model, mirror=True, match=symbol)

        # check to make sure all models downloaded
        if self.n_models_local(symbol) < self.n_models or overwrite_all:
            overwrite_all = True
            startdate = self.d_lower  # download full OHLC history
        else:
            offset = {1: 16, 15: 6}[interval]  # get days offset
            startdate = f.inter_now(interval) + delta(days=-offset)  # only download last 6 days

        # azure limited to 1.5gb memory
        if cf.AZURE_WEB and overwrite_all:
            raise RuntimeError('Cannot overwrite_all on azure.')

        log.info(f'fit_save_models: symbol={symbol}, overwrite={overwrite_all}, startdate={startdate}')

        n_periods = cf.dynamic_cfg(symbol=symbol, keys='target_n_periods')

        # FIXME temp solution
        exch_name = symbol.exch_name if not symbol == cf.MULTI_ALTS else 'bybit'

        # FIXME funding_exch
        # won't need to download all data necessarily
        df = self.dm.get_df(
            symbols={exch_name: symbol},
            startdate=startdate,
            interval=interval,
            # funding_exch=em.default('bitmex'),
            db_only=True,
            # local_only=not cf.AZURE_WEB
        )

        # laod df_close full history (~12s, 16mb, >1.25M rows)
        # TODO figure out how to cache the quantile, or compute from db?
        df_close = self.dm.get_df(
            symbols={exch_name: symbol},
            startdate=self.d_lower,
            interval=interval,
            db_only=True,
            close_only=True)

        wm = WeightsManager.from_config(df=df_close, symbol=symbol)

        # set back @cut_hrs due to losing @n_periods for training preds
        cut_hrs = math.ceil(n_periods / {1: 1, 15: 4}[self.interval])
        reset_hour_offset = self.reset_hour - cut_hrs  # 18
        # print('cut_hrs:', cut_hrs)  # 8
        # print('reset_hour_offset:', reset_hour_offset)  # 10

        # max date where hour is greater or equal to 18:00
        d_upper = f.date_to_dt(
            df.query('timestamp.dt.hour >= @reset_hour_offset')
            .index.get_level_values('timestamp').max().date()) \
            + delta(hours=reset_hour_offset)  # get date only eg '2022-01-01' then add delta hoursg
        # print('d_upper:', d_upper)  # 2022-01-17 10:00:00

        filter_quantile = cf.dynamic_cfg(symbol=symbol, keys='filter_fit_quantile')

        # add signals per symbol group, drop last targets
        # TODO add funding_exch: funding_exch=em.default('bitmex')
        df = df \
            .pipe(
                md.add_signals,
                name=name,
                symbol=symbol,
                use_important_dynamic=True,
                drop_ohlc=True,
                drop_target_periods=n_periods) \
            .pipe(
                wm.filter_quantile,
                quantile=filter_quantile,
                _log=False)

        # delete oldest model or all models
        if self.d_latest_model(symbol=symbol) < d_upper or overwrite_all:
            self.clean(symbol=symbol, last_only=not overwrite_all)

        cut_mins = {1: 60, 15: 15}[self.interval]  # to offset .loc[:d] for each prev model/day
        model = md.make_model(name, symbol=symbol)
        n_models = self.n_models

        # loop models from oldest > newest
        for i in range(n_models - 1, -1, -1):
            # log.warning(i)
            delta_mins = -i * cut_mins * self.batch_size_cdls
            d = d_upper + delta(minutes=delta_mins)  # cut 0, 24, 48 hrs
            d_cur_model = d + delta(hours=cut_hrs)
            d_prev = d + delta(hours=-self.batch_size)  # prev model date
            d_prev_model = d_prev + delta(hours=cut_hrs)
            # print('d:', d, 'd_prev:', d_prev, 'd_prev_model', d_prev_model)

            # do we have a previous model
            p_prev = self.get_model_path(symbol, d_prev_model)
            p_cur = self.get_model_path(symbol, d_cur_model)
            print('p_prev:', p_prev.name)
            print('p_cur:', p_cur.name)

            is_first = i == n_models - 1
            if not p_cur.exists() or (is_first and overwrite_all):

                # use df filtered to dates greater than last trained model
                if is_first:
                    init_model = None
                    d_lower = df.index.get_level_values('timestamp').min()
                else:
                    init_model = self.load_model(p=p_prev)
                    d_lower = d_prev + f.inter_offset(interval)
                print(f'rng: {d_lower} - {d}')

                idx_slice = pd.IndexSlice[:, d_lower: d]
                x_train, y_train = sk.split(df.loc[idx_slice, :])
                log.info(
                    f'fitting model: x_train.shape={x_train.shape}, is_first={is_first}, init_model={init_model}')

                init_model = model.fit(
                    X=x_train,
                    y=y_train,
                    sample_weight=wm.weights.loc[x_train.index],
                    init_model=init_model)

                # save - add back cut hrs so always consistent
                self.save_model(model=model, symbol=symbol, d=d_cur_model)
            else:
                log.info('model exists, not overwriting.')

        # mirror saved models to azure blob
        self.bs.upload_dir(p=self.p_model, mirror=True, match=symbol)

    def df_pred_from_models(self, df: pd.DataFrame, name: str, symbol: str) -> pd.DataFrame:
        """Load all models, make iterative predictions
        - probas added in a bit of a unique way here, one model slice at a time instead of all at once

        Parameters
        ----------
        df : pd.DataFrame
            truncated df (~400 rows?) with signals added and OHLC cols dropped, for live trading
        name : str
            model name
        symbol : str

        Returns
        -------
        pd.DataFrame
            df with all predictions added
        """
        self.bs.download_dir(p=self.p_model, mirror=True, match=symbol)

        cfg = md.model_cfg(name)
        target = cfg['target']
        pred_dfs = []

        p_models = self.local_models(symbol=symbol)

        if len(p_models) == 0:
            raise RuntimeError(f'No saved models found at: {self.p_model}')

        for i, p in enumerate(p_models):

            # get model train date from filepath
            d = self.parse_date(p=p)

            # load estimator
            model = jfl.load_pickle(p)  # type: LGBMClsLog

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
                    model=model)

                idx = df_pred.index
                fmt = self.dt_format
                log.info(f'Adding preds: {len(df_pred):02}, {idx.min():{fmt}}, {idx.max():{fmt}}')

                pred_dfs.append(df_pred)

        return df.pipe(pu.left_merge, pd.concat(pred_dfs)) \
            .pipe(md.add_proba_trade_signal)

    def to_dict(self):
        return ('d_lower', 'n_models', 'batch_size', 'batch_size_cdls')


if __name__ == '__main__':
    # %load_ext memory_profiler
    # BLACKFIRE_CLIENT_ID = "3dfdee9e-5944-425c-b182-d5f6e1c99994"
    # BLACKFIRE_CLIENT_TOKEN = "14583fb0dd4b2d0160d21f3bf45d963986c4483ae91e69afc04899452663e67d"
    # BLACKFIRE_SERVER_ID = "3e730235-6bbc-4c94-bdb8-ad0abdc108f5"
    # BLACKFIRE_SERVER_TOKEN = "146bd58683d7b58d4d00baba06ecd7c711adc8fa350f8269e3cc9d5bb96d7a8b"

    # from blackfire import probe
    # probe.initialize(
    #     client_id=BLACKFIRE_CLIENT_ID,
    #     client_token=BLACKFIRE_CLIENT_TOKEN
    # )
    # probe.enable()

    from jambot import livetrading as live
    from jambot.tradesys.symbols import Symbols

    # from jambot.ml.storage import ModelStorageManager
    em = live.ExchangeManager()
    msm = ModelStorageManager(test=True)
    syms = Symbols()
    symbol = cf.MULTI_ALTS
    # symbol = syms.symbol('XBTUSD')
    msm.fit_save_models(em=em, symbol=symbol)  # %memit

    # probe.end()
