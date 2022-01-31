import copy
import inspect
import operator as opr
import sys
import warnings
from collections import defaultdict as dd
from collections import deque
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import (AwesomeOscillatorIndicator, KAMAIndicator,  # noqa
                         PercentageVolumeOscillator, ROCIndicator,
                         RSIIndicator, StochasticOscillator, StochRSIIndicator,
                         TSIIndicator, UltimateOscillator)
from ta.trend import (ADXIndicator, AroonIndicator, CCIIndicator,  # noqa
                      DPOIndicator, EMAIndicator, KSTIndicator, MassIndex,
                      PSARIndicator, STCIndicator, TRIXIndicator)
from ta.volatility import UlcerIndex  # noqa
from ta.volume import AccDistIndexIndicator  # noqa
from ta.volume import MFIIndicator  # noqa
from ta.volume import ChaikinMoneyFlowIndicator

from jambot import SYMBOL
from jambot import config as cf
from jambot import getlog
from jambot import sklearn_utils as sk
from jambot.common import DictRepr, DynConfig
from jambot.config import AZURE_WEB
from jambot.utils.mlflow import MlflowLoggable
from jgutils import fileops as jfl
from jgutils import functions as jf
from jgutils import pandas_utils as pu
from jgutils.azureblob import BlobStorage

log = getlog(__name__)

warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')

# TODO distance to bolinger bands!


class SignalManager(MlflowLoggable, DictRepr):

    def __init__(
            self,
            target: str = None,
            slope: int = 0,
            sum: int = 0):

        self.signal_groups = {}  # type: Dict[str, SignalGroup]
        self.final_feats = []  # type: List[str]
        self.slope = slope
        self.sum = sum
        self.cut_periods = 0

        # cant add any new cols, but CAN exclude cols without pushing new code
        self.bs = BlobStorage(container=cf.p_data / 'feats')

    @classmethod
    def default(cls, **kw) -> 'SignalManager':
        """Load signalmanager with global default vals"""
        return cls(**cf.SIGNALMANAGER_KW, **kw)

    def to_dict(self) -> dict:
        return dict(signals=[sg for sg in self.signal_groups.values()])

    @property
    def log_items(self) -> dict:
        """Merge all logable params from signal groups"""
        m = {}
        for sg in self.signal_groups.values():
            # if isinstance(sg, MlflowLoggable):
            if hasattr(sg, 'log_items'):
                m |= sg.log_items

        return m | dict(
            num_feats=len(self.final_feats),
            feats_hash=self.feats_hash)

    @property
    def feats_hash(self) -> str:
        import hashlib
        import uuid

        seed = ''.join(self.final_feats)

        m = hashlib.md5()
        m.update(seed.encode('utf-8'))
        uid = uuid.UUID(m.hexdigest())
        return f'{len(self.final_feats)}_{uid}'

    def log_artifact(self) -> Path:
        """Save sm used columns as artifact for mlflow"""
        p = jfl.save_pickle(self.final_feats, cf.p_data / 'feats', 'feats')
        return p

    def get_signal_group(self, feature_name: str) -> 'SignalGroup':
        """Return signal_group object from feature_name"""
        return list(filter(lambda x: x.has_feature(feature_name), self.signal_groups.values()))[0]

    def print_signal_params(self, **kw):
        m = self.get_signal_params(**kw)
        jf.pretty_dict(m)

    def get_signal_params(self, feature_names=None, **kw) -> dict:
        """Return all optimizeable params from each signal feature"""
        m = {}

        # this is kinda messy
        for name, signal_group in self.signal_groups.items():
            for feature_name, m2 in (signal_group.signals or {}).items():
                if 'params' in m2:
                    if feature_names is None or feature_name in feature_names:
                        m[feature_name] = m2['params']
        return m

    def get_feature_names(self, *args, **kw):
        log.info('get_feature_names')
        return self.cols

    @property
    def lst_imp(self) -> List[str]:
        """NOTE bit messy, unfinished func"""
        if not hasattr(self, '_lst_imp'):
            self._lst_imp = jfl.load_pickle(p=cf.p_data / 'feats/most_imp_cols.pkl')

        return self._lst_imp

    def filter_n_feats(self, df: pd.DataFrame, n: int, lst_imp: List[str] = None) -> pd.DataFrame:

        if lst_imp is None:
            lst_imp = self.lst_imp

        # filter df to n most important feats
        cols = lst_imp[:n]

        # set final feats
        self.final_feats = cols

        default_cols = [c for c in cf.DROP_COLS if c in df.columns]
        return df[default_cols + cols + ['target']]

    def init_signal_group(self, signal_group: Union[str, 'SignalGroup']) -> 'SignalGroup':
        """Instantiate signal group if str, save to self.signal_groups dict

        Returns
        -------
        SignalGroup
            instantiated SignalGroup
        """

        # if str, init obj with defauls args
        if isinstance(signal_group, str):
            signal_group = getattr(sys.modules[__name__], signal_group)()  # type: SignalGroup

        # save to dict of self.signal_groups
        self.signal_groups[signal_group.class_name] = signal_group
        return signal_group

    def add_signals(
            self,
            df: pd.DataFrame,
            signals: List[Union[str, Any]],
            signal_params: dict = None,
            use_important: bool = False,
            exclude_cols: List[str] = None,
            n_most_imp: int = None,
            use_important_dynamic: bool = False,
            drop_ohlc: bool = False,
            symbol: str = SYMBOL,
            **kw) -> pd.DataFrame:
        """Add multiple initialized signals to dataframe"""
        signal_params = signal_params or {}
        final_signals = {}
        drop_cols = []
        require_cols = {}  # from signal_group, dict of all cols w their requirements
        cut_periods = self.cut_periods
        _signal_groups = []  # type: List[SignalGroup]

        # Loop SignalGroup objs (not single columns)
        for signal_group in jf.as_list(signals):

            signal_group = self.init_signal_group(signal_group)
            _signal_groups.append(signal_group)
            cut_periods = max(cut_periods, signal_group.cut_periods)  # TODO dont want cached sigs to cut again
            signal_group.df = df  # NOTE kinda sketch
            signal_group.slope = self.slope  # sketch too
            signal_group.sum = self.sum

            # force_overwrite=False
            final_signals |= signal_group.make_signals(df=df, **kw)
            del signal_group.df
            drop_cols += signal_group.drop_cols
            require_cols |= signal_group.require_cols

        if use_important_dynamic:
            n_most_imp = cf.dynamic_cfg(symbol=symbol)['num_feats']

        # filter only most imporant cols before adding all
        if use_important or n_most_imp:
            processed, dep_only = [], []
            # TODO THIS NEEDS TO STAY IN SYNC WITH LIVE DATA IF LIVE TRADING
            # if AZURE_WEB or True:
            if AZURE_WEB:
                prefix = 'most' if n_most_imp else 'least'
                self.bs.download_file(p=f'{prefix}_imp_cols.pkl')

            if not n_most_imp:
                # EXCLUDE least_imp_cols
                if exclude_cols is None:
                    exclude_cols = jfl.load_pickle(p=self.bs.p_local / 'least_imp_cols.pkl')  # type: List[str]

                if not exclude_cols:
                    raise RuntimeError('No cols in exclude_cols')

                include_cols = [c for c in final_signals.keys() if not c in exclude_cols]
            else:
                # INCLUDE most_imp_cols
                # NOTE adding target this way messy
                include_cols = jfl.load_pickle(p=self.bs.p_local / 'most_imp_cols.pkl')[:n_most_imp] + ['target']

            q = deque(include_cols)

            # recursively handle identify cols which important cols are dependent on
            while q:
                col = q.pop()

                # get list of columns that this col depends on, add to queue
                dep_cols = require_cols.get(col, None)
                if dep_cols:
                    q.extend([c for c in jf.as_list(dep_cols) if not c in processed])

                # final cols which will be included
                if not col in processed:
                    processed.append(col)

                    # cols which will be dropped after
                    if not col in include_cols:
                        dep_only.append(col)

            final_signals = {k: v for k, v in final_signals.items() if k in processed}
            drop_cols.extend(dep_only)

        # save for logging
        self.final_feats = sorted(list(
            set(list(final_signals.keys()) + self.final_feats) - set(drop_cols)))

        # print('len final_feats:', len(self.final_feats))
        # print('len final_signals:', len(final_signals))

        # NOTE not sure if we should always fillna(0) here, maybe force errors and confront
        df = df.assign(**final_signals) \
            .pipe(pu.safe_drop, cols=drop_cols) \
            .iloc[cut_periods:, :] \
            .fillna(0) \
            .pipe(pu.safe_drop, cols=cf.DROP_COLS, do=drop_ohlc)

        self.save_cache_cols(df, _signal_groups)

        return df

    def save_cache_cols(self, df: pd.DataFrame, signal_groups: List['SignalGroup']) -> None:
        """Cache cols to avoid recalc later

        Parameters
        ----------
        df : pd.DataFrame
            df with all signal cols
        signal_groups : List[SignalGroup]
        """
        for sg in signal_groups:
            # don't save again if already saved
            if sg.cache and not sg.cached_cols:
                # log.info(f'saving cached cols')
                sg.cached_cols = {c: df[c] for c in sg.cache_cols_names}

    def show_signals(self):
        m = dd(dict)
        for name, signal_group in self.signal_groups.items():

            for sig_name, params in signal_group.signals.items():
                if 'cls' in params:
                    m[name].update({sig_name: params['cls'].__name__})
                else:
                    m[name].update({sig_name: ''})

        sk.pretty_dict(m)

    def replace_single_feature_old(
            self,
            df: pd.DataFrame,
            feature_params: dict,
            **kw) -> Iterable[Tuple[pd.DataFrame, str]]:
        """Generator to replace single feature at a time with new values from params dict"""

        for feature_name, m in feature_params.items():
            # get signal group from feature_name
            signal_group = self.get_signal_group(feature_name=feature_name)
            feature_name = feature_name.replace(f'{signal_group.prefix}_', '')

            for param, vals in m.items():
                for val in vals:
                    param_single = {feature_name: {param: val}}  # eg {'rsi': {'window': 12}}

                    yield df.pipe(
                        signal_group.make_signals,
                        params=param_single,
                        force_overwrite=True), f'{feature_name}_{param}_{val}'

    def make_plot_traces(self, name) -> list:
        """Create list of dicts for plot traces"""
        name = name.lower()
        signal_group = self.signal_groups[name].signals

        return [dict(name=name, func=m['plot'], row=m.get('row', None)) for name, m in signal_group.items()]

    def plot_signal_group(self, name, **kw):
        """Convenience func to show plot with a full group of signals"""
        from jambot import charts as ch
        fig = ch.chart(
            df=self.df,
            traces=self.make_plot_traces(name=name),
            **kw)

        self.fig = fig
        fig.show()


class SignalGroup(DictRepr):
    """Base class for ta indicators"""
    # Default OHLCV cols for df, used to pass to ta signals during init
    m_default = dict(
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume')

    def __init__(
            self,
            # df: pd.DataFrame = None,
            signals: Dict[str, Any] = None,
            fillna: bool = True,
            prefix: str = None,
            cache: bool = False,
            **kw):
        """Obj to create one or multiple signal columns

        Parameters
        ----------
        signals : Dict[str, Any], optional
            dict of {signal name: func}, by default None
        fillna : bool, optional
            default True
        prefix : str, optional
            add prefix to individual signal names, default None
        cache : bool, optional
            cache self.cache_cols, by default False
        """
        self.drop_cols = []
        self.no_slope_cols = []
        self.no_sum_cols = []
        self.not_normal_cols = []
        self.force_slope_cols = []
        self.require_cols = {}

        # self.df = df
        self.prefix = prefix
        self.fillna = fillna
        self.signals = self.init_signals(signals)
        self.slope = 0
        self.sum = 0
        self.class_name = self.__class__.__name__.lower()
        self.cut_periods = 0  # drop n rows at start of df
        self.cache = cache
        self.cache_cols_names = []  # type: List[str]
        self.cached_cols = {}  # type: Dict[str, pd.Series]

    def to_dict(self) -> dict:
        return dict(num_cols=len(self.signals))

    def init_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Extra default processing to signals, eg add plot func"""
        # if signals is None:
        #     return

        for name, m in signals.items():

            if not isinstance(m, dict):
                if callable(m):
                    m = dict(func=m)  # signal was just created as single func
                else:
                    raise AttributeError('signal must be init with a dict or callable func!')

            m['name'] = name

            # assign scatter as default plot
            # if m.get('plot', None) is None:
            #     m['plot'] = ch.scatter

            signals[name] = m

        return signals

    def has_feature(self, feature_name: str) -> bool:
        """Check if SignalGroup contains requested feature name
        - will check if has prefix too

        Parameters
        ----------
        feature_name : str
            feature to check for

        Returns
        -------
        bool
            if has feature or not
        """
        feature_name = feature_name.replace(f'{self.prefix}_', '')
        return feature_name in self.signals

    def add_all_signals(self, df: pd.DataFrame, **kw) -> pd.DataFrame:
        """Convenience base wrapper to work with ta/non-ta signals
        NOTE not used with SignalManager now, but could be used for individual SignalGroup
        """
        return df \
            .assign(**self.make_signals(df=df)) \
            .pipe(pu.safe_drop, cols=self.drop_cols)

    def make_signals(
            self,
            df: pd.DataFrame,
            params: dict = None,
            names: list = None,
            force_overwrite: bool = False) -> dict:
        """loop self.signal_groups, init, call correct func and assign to df column

        Parameters
        ----------
        params : dict
            if params passed, only reassign those features (for quick optimization testing)
            params must be dict of {feature_name: {single param: value}}
        names : list
            init only specific signals by name
        force_overwrite : bool, default false
            overwrite final signals (for optimization testing)

        Returns
        -------
        dict
            dict of final signals to be used with .assign()
        """

        # return initialized cached series instead of func to create series
        if self.cache and self.cached_cols:
            # log.info('returing cached cols')
            return self.cached_cols

        final_signals = {}
        signals = self.signals if params is None else params

        # filter specific names to init specific signals only
        if not names is None:
            signals = {k: v for k, v in signals.items() if k in jf.as_list(names)}

        for feature_name, m in signals.items():

            if not params is None:
                # overwrite original feature params with new value eg {'mnt_rsi': {'window': 12}}
                m = {**self.signals[feature_name], **m}

            if 'cls' in m:
                # init ta signal
                final_signals |= self.make_signal(**m)
                # final_signals[feature_name] = self.make_signal(**m)
            elif 'func' in m:
                final_signals[feature_name] = m['func']
            else:
                raise AttributeError('No func to assign from signal!')

        # rename cols with prefix_
        if self.prefix:
            final_signals = {f'{self.prefix}_{k}': v for k, v in final_signals.items()}

        # dont add slope for any Target classes, drop_cols, or no_slope_cols
        if self.slope and not 'target' in self.class_name:
            final_signals |= self.add_slope(signals=final_signals)

        # NOTE coud be more dry
        if self.sum and not 'target' in self.class_name:
            final_signals |= self.add_sum(signals=final_signals)

        # filter out signals that already exist in df
        if not force_overwrite:
            final_signals = {k: v for k, v in final_signals.items() if not k in df.columns}

        return final_signals

    def update_deps(self, m: dict, prefix: str) -> None:
        """Add original cols as dependency of new col with prefix

        Parameters
        ----------
        m : dict
            dict to add deps of
        prefix : str
            prefix to remove in dependent col
        """
        m_update = {k: k.replace(prefix, '') for k in m.keys()}
        self.require_cols |= m_update

    def add_slope(self, signals: dict) -> dict:
        """Create dict of slope siglals for all input signals
        - add simple rate of change for all signal columns

        Parameters
        ----------
        signals : dict
            signals to calc slope on

        Returns
        -------
        dict
            slope signals
        """
        exclude = list(set(self.drop_cols + self.no_slope_cols) - set(self.force_slope_cols))
        m_dy = {}

        for n_periods in jf.as_list(self.slope):
            prefix = f'dy{n_periods:02}_'

            m_temp = {
                f'{prefix}{c}': lambda x, c=c, n_periods=n_periods: self.make_slope(
                    s=x[c],
                    n_periods=n_periods,
                    normal_slope=not c in self.not_normal_cols) for c in signals if not c in exclude}

            self.update_deps(m_temp, prefix=prefix)
            m_dy |= m_temp

        return m_dy

    def add_sum(self, signals: dict) -> dict:
        """Create dict of sum siglals for all input signals

        Parameters
        ----------
        signals : dict
            signals to calc slope on

        Returns
        -------
        dict
            slope signals
        """
        exclude = self.drop_cols + self.no_sum_cols
        m_sum = {}

        for n_periods in jf.as_list(self.sum):
            prefix = f'sm{n_periods:02}_'

            m_temp = {
                f'{prefix}{c}': lambda x, c=c, n_periods=n_periods:
                    x[c].rolling(n_periods).sum().astype(np.float32) for c in signals if not c in exclude}

            self.update_deps(m_temp, prefix=prefix)
            m_sum |= m_temp

        return m_sum

    @property
    def default_cols(self) -> Dict[str, pd.Series]:
        """Convert default dict of strings to pd.Series for default cols"""
        df = self.df
        if df is None:
            raise AttributeError('Need to set df first!')

        return {name: df[col] for name, col in self.m_default.items()}

    def filter_valid_kws(self, cls, **kw) -> dict:
        """Check default args of __init__ + merge with extra kws, return dict of valid kws"""
        signature = inspect.signature(cls.__init__)

        # merge extra_kws with default OHLC cols
        m_all_kw = {**self.default_cols, **(kw or {})}

        return {name: col for name, col in m_all_kw.items() if name in signature.parameters.keys()}

    def make_signal(self, cls, name: str, ta_func: str, **kw):
        """Helper func to init TA signal with correct OHLCV columns

        Parameters
        ---------
        name : str
            name for dict key to return
        cls : ta
            class defn of ta indicator to be init
        ta_func : str
            func to call on ta obj to create final signal

        Returns
        -------
        dict : dict of {name: ta signal column}
        """

        # most will be single string, but allow list of multiple
        ta_func = jf.as_list(ta_func)

        kw['fillna'] = kw.get('fillna', self.fillna)
        good_kw = self.filter_valid_kws(cls=cls, **kw)

        # init ta obj
        signal = cls(**good_kw)

        # allow adding multiple columns from single ta signal obj
        m = {}
        for func in ta_func:
            key = f'{name}_{func}' if len(ta_func) > 1 else name

            # use dict as ordered set here
            key = '_'.join({k: 1 for k in key.split('_')}.keys())  # remove duplicates for multi cols per signal
            m[key] = lambda x: getattr(signal, func)().astype(np.float32)

        return m

    def make_slope(self, s: pd.Series, n_periods: int = 1, normal_slope: bool = True) -> pd.Series:
        """Return series as slope of input series
        - NOTE - pct_change returns inf when previous value is 0

        Parameters
        ----------
        s : pd.Series
            input series
        n_periods : int, optional
            number periods to calc slope over, default 1
        normal_slope : bool, optional
            is series normalized (use actual change, not %), default True

        Returns
        -------
        pd.Series
            slope of series
        """
        # return (s - np.roll(s, n_periods, axis=0)) / n_periods
        if not normal_slope:
            return s.pct_change(n_periods).astype(np.float32)
        else:
            return (s.diff(n_periods) / n_periods).astype(np.float32)


class FeatureInteraction(SignalGroup):
    """Calc differenc btwn two columns"""

    def __init__(self, cols, int_type='sub', **kw):
        kw['signals'] = dict()

        super().__init__(**kw)

        op = dict(
            sub=np.subtract,
            add=np.add,
            mul=np.multiply,
            div=np.divide) \
            .get(int_type)

        jf.set_self()

    def add_all_signals(self, df, **kw):
        cols = self.cols
        scaler = MinMaxScaler()
        cols_scaled = scaler.fit_transform(df[cols])
        cols_int = self.op(cols_scaled.T[0], cols_scaled.T[1])

        # name = f'{cols[0]}_{cols[1]}_diff'
        return df.assign(**{self.int_type: cols_int})


class Momentum(SignalGroup):
    def __init__(self, window=2, **kw):
        kw['signals'] = dict(
            pvo=dict(cls=PercentageVolumeOscillator, ta_func='pvo', window_slow=26, window_fast=12, window_sign=9),
            roc=dict(cls=ROCIndicator, ta_func='roc', window=12),  # Rate of Change (similar to pct_change?)
            stoch=dict(cls=StochasticOscillator, ta_func='stoch', window=12, smooth_window=12),
            tsi=dict(cls=TSIIndicator, ta_func='tsi', window_slow=25, window_fast=13),
            ultimate=dict(
                cls=UltimateOscillator,
                ta_func='ultimate_oscillator',
                window1=7, window2=14, window3=28,
                weight1=4.0, weight2=2.0, weight3=1.0),
            # awesome=dict(
            #     cls=AwesomeOscillatorIndicator,
            #     ta_func='awesome_oscillator',
            #     window1=10,
            #     window2=50,
            #     params=dict(
            #         window1=[6, 12, 18],
            #         window2=[36, 50, 200])),
            # awesome_rel=lambda x: x.mnt_awesome / x.ema50
            # kama=dict(cls=KAMAIndicator, ta_func='kama', window=12, pow1=2, pow2=30, row=1)
        )
        ns = (2, 6, 12, 24)
        m_rsi = {f'rsi{n:02}': dict(cls=RSIIndicator, ta_func='rsi', window=n) for n in ns}
        kw['signals'] |= m_rsi

        super().__init__(prefix='mnt', **kw)
        # drop_cols = ['awesome']
        jf.set_self()


class Volume(SignalGroup):
    def __init__(self, **kw):
        prefix = 'vol'
        kw['signals'] = dict(
            # relative=lambda x: x.volume / x.volume.shift(6).rolling(24).mean(),
            rel=lambda x: relative_self(x.volume, n=24 * 7),
            # mom=lambda x: x.vol_relative * x.pct,
            chaik=dict(cls=ChaikinMoneyFlowIndicator, ta_func='chaikin_money_flow', window=24),
            mfi=dict(cls=MFIIndicator, ta_func='money_flow_index', window=48),
            # adi=dict(cls=AccDistIndexIndicator, ta_func='acc_dist_index'),
            # eom=dict(cls=EaseOfMovementIndicator, ta_func='ease_of_movement', window=14),
            # force=dict(cls=ForceIndexIndicator, ta_func='force_index', window=14),
        )

        super().__init__(**kw)
        jf.set_self()


class Volatility(SignalGroup):
    def __init__(self, norm=(0.004, 0.024), **kw):
        prefix = 'vty'
        kw['signals'] = dict(
            ulcer=dict(cls=UlcerIndex, ta_func='ulcer_index', window=2),
            maxhigh=lambda x: x.high.rolling(48).max(),
            minlow=lambda x: x.low.rolling(48).min(),
            spread=lambda x: (abs(x.vty_maxhigh - x.vty_minlow) /
                              x[['vty_maxhigh', 'vty_minlow']].mean(axis=1)).astype(np.float32),
            ema=lambda x: x.vty_spread.ewm(span=60, min_periods=60).mean().astype(np.float32),
            sma=lambda x: x.vty_spread.rolling(300).mean().astype(np.float32),
            # norm_ema=lambda x: np.interp(x.vty_ema, (0, 0.25), (norm[0], norm[1])),
            # norm_sma=lambda x: np.interp(x.vty_sma, (0, 0.25), (norm[0], norm[1])),
        )
        ns = (48, 96, 192, 384)
        m_rng = {f'rng{n:03}': lambda x, n=n: self.vty_range(x, n_periods=n) for n in ns}
        kw['signals'] |= m_rng

        super().__init__(**kw)
        no_slope_cols = [f'vty_{c}' for c in m_rng]
        drop_cols = ['spread', 'vty_maxhigh', 'vty_minlow']
        require_cols = dict(
            vty_spread=['vty_maxhigh', 'vty_minlow'],
            vty_ema='vty_spread',
            vty_sma='vty_spread')
        jf.set_self()

    @staticmethod
    def vty_range(df: pd.DataFrame, n_periods: int = 48) -> pd.Series:

        # could use candle bodys only or high/low
        rng_high = df.close.rolling(n_periods).max()
        rng_low = df.close.rolling(n_periods).min()
        rng = rng_high - rng_low
        cdl_size = np.abs(df.close - df.open).rolling(n_periods).sum()
        return (rng / cdl_size).astype(np.float32)


class Trend(SignalGroup):
    def __init__(self, **kw):
        kw['signals'] = dict(
            adx=dict(cls=ADXIndicator, ta_func='adx', window=36),
            aroon=dict(cls=AroonIndicator, ta_func='aroon_indicator', window=25),
            cci=dict(cls=CCIIndicator, ta_func='cci', window=20, constant=0.015),
            # mass=dict(cls=MassIndex, ta_func='mass_index', window_fast=9, window_slow=25),
            stc=dict(cls=STCIndicator, ta_func='stc', window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3),
            # dpo=dict(cls=DPOIndicator, ta_func='dpo'),
            # kst=dict(cls=KSTIndicator, ta_func='kst'),
            # trix=dict(cls=TRIXIndicator, ta_func='trix', window=48),
            # psar=dict(cls=PSARIndicator, ta_func=['psar', 'psar_down', 'psar_up'],
            # # step=0.02, max_step=0.2, fillna=False)

        )

        super().__init__(prefix='trend', **kw)
        # accept 1 to n series of trend signals, eg 1 or -1
        jf.set_self()


class EMA(SignalGroup):
    """Add pxhigh, pxlow for rolling period, depending if ema_trend is positive or neg

    w = with, a = agains, n = neutral (neutral not used)
    """

    def __init__(self, fast: int = 50, slow: int = 200, **kw):

        # against, wth, neutral = speed[0], speed[1], int(np.average(speed))
        colfast, colslow = f'ema{fast}', f'ema{slow}'
        c = self.get_c(maxspread=0.1)

        emas = [10, 50, 200]
        m_emas = {f'ema{n}': dict(cls=EMAIndicator, ta_func='ema_indicator', window=n) for n in emas}
        kw['signals'] = copy.copy(m_emas)

        kw['signals'] |= dict(
            ema_spread=lambda x: (x[colfast] - x[colslow]) / ((x[colfast] + x[colslow]) / 2),
            ema_conf=lambda x: self.ema_exp(s=x.ema_spread, c=c),
        )

        super().__init__(**kw)

        self.drop_cols = list(m_emas)
        self.force_slope_cols = list(m_emas)
        self.not_normal_cols = list(m_emas)
        self.no_slope_cols = ['ema_trend']
        self.cut_periods = max(emas)  # drop first 200 rows after adding

        self.require_cols = dict(
            ema_spread=[colfast, colslow],
            ema_conf='ema_spread')

    def get_c(self, maxspread: float) -> float:
        """C coefficient for use in ema_exp"""
        m = -2.9
        b = 0.135
        return round(m * maxspread + b, 2)

    def ema_exp(self, s: pd.Series, c: float) -> np.ndarray:
        """Sigmoid ish func to squash difference btwn emas"""
        side = np.where(s >= 0, 1, -1)
        x = abs(s)

        aLim = 2
        a = -1000
        b = 3
        d = -3
        g = 1.7

        y = side * (a * x ** b + d * x ** g) / (aLim * a * x ** b + aLim * d * x ** g + c)

        return y.astype(np.float32)


class DateTime(SignalGroup):
    def __init__(self, **kw):
        kw['signals'] = dict(
            day_week=lambda x: x.index.dayofweek,
            time_day=lambda x: x.index.hour)

        super().__init__(**kw)
        no_slope_cols = list(kw['signals'].keys())
        no_sum_cols = no_slope_cols

        jf.set_self()


class MACD(SignalGroup):
    """MACD signals
    - NOTE not used currently
    """

    def __init__(self, fast=50, slow=200, smooth=50, **kw):

        kw['signals'] = dict(
            macd=lambda x: x[f'ema{fast}'] - x[f'ema{slow}'],
            macd_signal=lambda x: x.macd.ewm(span=smooth, min_periods=smooth).mean(),
            macd_diff=lambda x: x.macd - x.macd_signal,
            macd_trend=lambda x: np.where(x.macd_diff > 0, 1, -1)
        )

        super().__init__(**kw)
        name = 'macd'
        trendseries = 'macd_trend'
        no_slope_cols = ['macd_trend']
        jf.set_self()

    def add_all_signals(self, df):
        return df \
            .pipe(add_emas, emas=[self.fast, self.slow]) \
            .pipe(super().add_all_signals)

    def final(self, side, c):
        conf = 1.25 if side * c.macd_trend == 1 else 0.5
        return conf * self.weight


class SFP(SignalGroup):
    """
    Calculate wether candle is sfp, for any of 3 previous max/min periods
    - This signal needs cdl body signals init first
    - NOTE could maybe use min/max peaks from tsfresh?
    """

    def __init__(self, period_base=48, **kw):
        super().__init__(**kw)
        minswing = 0.05  # NOTE could be hyperparam
        jf.set_self()

    def add_all_signals(self, df):

        # set min low/max high for each rolling period
        offset = 6
        periods = [self.period_base * 2 ** i for i in range(3)]

        m = dict(high=['max', 1], low=['min', -1])

        for period in periods:
            for extrema, v in m.items():
                agg_func, side = v[0], v[1]
                check_extrema = f'{extrema.lower()}_{period}'  # high_48

                s = df[extrema].rolling(period)

                df[check_extrema] = getattr(s, agg_func)().shift(offset)  # min() or max()

                # calc candle is swing fail or not
                # minswing means tail must also be greater than min % of candle full size
                df[f'sfp_{check_extrema}'] = np.where(
                    (side * (df[extrema] - df[check_extrema]) > 0) &
                    (side * (df.close - df[check_extrema] < 0)) &
                    (df[f'cdl_tail_{extrema.lower()}'] / df.cdl_full > self.minswing),
                    1, 0)

        return df \
            .pipe(pu.drop_cols, expr=r'^(high|low)_\d+') \
            .assign(**self.add_sum(signals=pu.filter_cols(df, 'sfp')))


class Candle(SignalGroup):
    """
    - cdl_side > wether candle is red or green > (open - close)
    - cdl_full > size of total candle (% high - low) > relative to volatility
    - cdl_body > size of candle body (% close - open)
    - tail_size_low, tail_size_high > size of tails (% close - low, high - low) > scale to volatility?
    - 200_v_high, 200_v_low > % proximity of candle low/high to 200ema
        - close to support with large tail could mean reversal
    - high_above_prevhigh, close_above_prevhigh > maybe add in trend? similar to close_v_range
    - low_below_prevlow, close_below_prevlow
    - close_v_range > did we close towards upper or lower side of prev 24 period min/max range
    """

    def __init__(self, **kw):
        # TODO #5
        n_periods = 24  # used to cal relative position of close to prev range
        kw['signals'] = dict(
            pct=lambda x: x.close.pct_change(24),
            # min_n=lambda x: x.low.rolling(n_periods).min(),
            # range_n=lambda x: (x.high.rolling(n_periods).max() - x.min_n),
            # cdl_side=lambda x: np.where(x.close > x.open, 1, -1).astype(np.int8),
            cdl_full=lambda x: (x.high - x.low) / x.open,
            cdl_body=lambda x: (x.close - x.open) / x.open,
            cdl_full_rel=lambda x: relative_self(x.cdl_full, n=24),
            cdl_body_rel=lambda x: relative_self(x.cdl_body, n=24),
            cdl_tail_high=lambda x: np.abs(x.high - x[['close', 'open']].max(axis=1)) / x.open,
            cdl_tail_low=lambda x: np.abs(x.low - x[['close', 'open']].min(axis=1)) / x.open,
            ema200_v_high=lambda x: np.abs(x.high - x.ema200) / x.open,
            ema200_v_low=lambda x: np.abs(x.low - x.ema200) / x.open,
            # pxhigh=lambda x: x.high.rolling(n_periods).max().shift(1),
            # pxlow=lambda x: x.low.rolling(n_periods).min().shift(1),
            # high_above_prevhigh=lambda x: np.where(x.high > x.pxhigh, 1, 0).astype(bool),
            # close_above_prevhigh=lambda x: np.where(x.close > x.pxhigh, 1, 0).astype(bool),
            # low_below_prevlow=lambda x: np.where(x.low < x.pxlow, 1, 0).astype(bool),
            # close_below_prevlow=lambda x: np.where(x.close < x.pxlow, 1, 0).astype(bool),
            # buy_pressure=lambda x: (x.close - x.low.rolling(2).min().shift(1)) / x.close,
            # sell_pressure=lambda x: (x.close - x.high.rolling(2).max().shift(1)) / x.close,
        )

        # close v range
        ns = (24, 48, 96, 192)
        # m_cvr = {f'cvr{n:03}': lambda x, n=n: self.close_v_range(x, n_periods=n) for n in ns}
        m_cvr = {}
        m_brk = {f'brk{n:03}': lambda x, n=n: self.breakout(x, n_periods=n) for n in ns}

        # ns = (6, 12, 24)
        # m_cdl_body = {f'cdl_body_rel_{n}': lambda x, n=n: relative_self(x.cdl_body, n=n) for n in ns}

        kw['signals'] |= m_cvr | m_brk  # | m_cdl_body

        super().__init__(**kw)
        drop_cols = ['min_n', 'range_n', 'pxhigh', 'pxlow']
        no_slope_cols = [
            'cdl_side',
            'cdl_full',
            'cdl_body',
            'high_above_prevhigh',
            'close_above_prevhigh',
            'low_below_prevlow',
            'close_below_prevlow',
            'pxhigh',
            'pxlow',
            'cdl_tail_low',
            'cdl_tail_high']  # + list(m_brk.keys())
        # no_sum_cols = list(m_brk.keys())

        require_cols = dict(
            cdl_full_rel='cdl_full',
            cdl_body_rel='cdl_body',
            ema200_v_high='ema200',
            ema200_v_low='ema200',
            high_above_prevhigh='pxhigh',
            close_above_prevhigh='pxhigh',
            low_below_prevlow='pxlow',
            close_below_prevlow='pxlow') \
            # | {k: 'cdl_body' for k in m_cdl_body}

        jf.set_self()

    @staticmethod
    def breakout(df: pd.DataFrame, n_periods: int, n_shift: int = 12) -> pd.Series:
        """Combination of cvr + extra "bump" when price brakes out of trailing range high/low

        Parameters
        ----------
        df : pd.DataFrame
        n_periods : int
            lookback period for range high/low
        n_shift : int, optional
            shift period for breakouts, by default 12

        Returns
        -------
        pd.Series
        """
        max_n = df.high.rolling(n_periods).max()
        max_n_shift = max_n.shift(n_shift)
        min_n = df.low.rolling(n_periods).min()
        min_n_shift = min_n.shift(n_shift)

        s = np.where(
            df.close > max_n_shift,
            df.close - max_n_shift,
            np.where(
                df.close < min_n_shift,
                df.close - min_n_shift, 0)) / df.close

        # cvr (faster, resuing rolling windows)
        cvr = (df.close - min_n) / (max_n - min_n)
        cvr = cvr.fillna(cvr.mean()) - 0.5

        # cols = ['high', 'low', 'close']
        # df[cols].iloc[-300:] \
        #     .assign(cvr=lambda x: sg.Candle.breakout(x, 96)) \
        #     ['cvr'].plot();

        # return (s * 10).astype(np.float32)
        # * 10 to put breakout on same scale as cvr
        return ((s * 10) + cvr).astype(np.float32)

    @staticmethod
    def close_v_range(df: pd.DataFrame, n_periods: int) -> pd.Series:

        min_n = df.low.rolling(n_periods).min()
        range_n = (df.high.rolling(n_periods).max() - min_n)
        s = ((df.close - min_n) / range_n)  # - 0.5
        return s.fillna(s.mean()).astype(np.float32)

    def _add_all_signals(self, df):
        # NOTE could be kinda wrong div by open, possibly try div by SMA?
        # TODO NEED candle body sizes relative to current rolling volatility

        # .pipe(lambda df: df.fillna(value=dict(
        #     buy_pressure=0,
        #     sell_pressure=0,
        #     # close_v_range=df.close_v_range.mean()
        # ))) \
        # .pipe(add_emas) \
        return df \
            .pipe(super().add_all_signals)


class CandlePatterns(SignalGroup):
    def __init__(self, **kw):
        import talib as tb
        cdl_names = tb.get_function_groups()['Pattern Recognition']

        kw['signals'] = {c.lower().replace('cdl', 'cdp_'): lambda x, c=c: getattr(tb, c)
                         (x.open, x.high, x.low, x.close) for c in cdl_names}

        super().__init__(**kw)

        # don't add slope for any cdl patterns
        no_slope_cols = list(kw['signals'])
        jf.set_self()


class TargetClass(SignalGroup, MlflowLoggable, DynConfig):
    """
    target classification
    - 3 classes
    - higher, lower, or same in x periods into future (10?)
    - same = (within x%, 0.5? > scale this based on daily volatility)
    - calc close price vs current price, x periods in future (rolling method)
    # NOTE doesn't work currently need to redo with kw['signals]
    """
    log_keys = dict(target_n_periods='n_periods')

    def __init__(self, n_periods: int = 10, **kw):
        super().__init__(**kw)
        self.n_periods = n_periods
        self.cache_cols_names = ['target']

    @property
    def log_items(self) -> Dict[str, Any]:
        return dict(target_n_periods=self.n_periods)

    def to_dict(self) -> dict:
        return dict(n_periods=self.n_periods)


class TargetMeanEMA(TargetClass):
    def __init__(self, p_ema: int = 10, **kw):
        # NOTE not finished, don't like ema
        predict_col = self.ema_col
        ema_col = f'ema{p_ema}'

        kw['signals'] = dict(
            pct_future=lambda x: (x[predict_col].shift(-self.n_periods) - x[predict_col]) / x[predict_col],
            target=lambda x: np.where(x.pct_future > 0, 1, -1))

        super().__init__(p_ema=p_ema, **kw)

        drop_cols = []
        jf.set_self()

    def add_all_signals(self, df):
        pct_min = self.pct_min  # NOTE this could be a hyperparam
        # TODO ^ definitely needs to be scaled to daily volatility

        predict_col = self.ema_col
        # predict_col = 'close'

        return df \
            .pipe(add_ema, p=self.p_ema) \
            .assign(
                pct_future=lambda x: (x[predict_col].shift(-self.n_periods) - x[predict_col]) / x[predict_col],
                target=lambda x: np.where(x.pct_future > pct_min, 1, np.where(x.pct_future < -pct_min, -1, 0))) \
            .drop(columns=['pct_future'])


class TargetMean(TargetClass):
    """
    Calc avg price of next n close prices
    - maybe - bin into % ranges?
    """

    def __init__(self, n_periods: int, regression: bool = False, pct_min: float = 0, **kw):

        if not regression:
            kw['signals'] = dict(
                target=lambda x: np.sign(
                    x.close.rolling(n_periods).mean().shift(-n_periods) - x.close))

        else:
            # just use % diff for regression instead of -1/1 classification
            kw['signals'] = dict(
                target=lambda x: (x.close.rolling(n_periods).mean().shift(-n_periods) - x.close) / x.close)

        super().__init__(**kw)

        jf.set_self()


class TargetMaxMin(TargetClass):
    """
    Create two outputs - min_low, max_high
    """

    def __init__(self, n_periods: int, use_close: bool = True, **kw):

        if use_close:
            items = [('max', 'close'), ('min', 'close')]
        else:
            items = [('max', 'high'), ('min', 'low')]

        kw['signals'] = {f'target_{fn}': lambda x, fn=fn, c=c: (
            x[c]
            .rolling(n_periods).__getattribute__(fn)()
            .shift(-n_periods) - x.close) / x.close for fn, c in items}

        super().__init__(n_periods=n_periods, **kw)


class TargetUpsideDownside(TargetMaxMin):
    """
    - NOTE last n_periods of target will be -1, NOT NaN > need to make sure to drop
    """

    def __init__(self, n_periods: int, **kw):
        super().__init__(n_periods=n_periods, **kw)

        self.signals |= dict(
            target=lambda x: np.where(
                x.target_max.notna(),
                np.where(np.abs(x.target_max) > np.abs(x.target_min), 1, -1),
                np.NaN).astype(np.int8)  # int8 converts NaN to 0
        )

        self.drop_cols = ['target_max', 'target_min']
        self.signals = self.init_signals(self.signals)
        self.require_cols = dict(target=self.drop_cols)


class TargetPercentile(TargetClass):
    """Is current close in upper or lower 50% of future n_periods
    lmao this is the EXACT same as TargetUpdsideDownside...
    """

    def __init__(self, n_periods: int, **kw):
        n = n_periods

        kw['signals'] = dict(
            target=lambda x: np.where(
                ((x.close.rolling(n).max() + x.close.rolling(n).min()).shift(-n) / 2) - x.close > 0, 1, -1)
        )

        super().__init__(n_periods=n_periods, **kw)


class TargetRatio(TargetClass):
    #  could either do ratio of up-downside
    #   OR difference % of up - downside
    #  using only close vs closes
    #    OR close vs maxhigh/minlow?

    # will need to define ratios for enter/exit
    # eg 10% or so middle band where enter into means close current trade, exit out means enter new

    def __init__(self, n_periods: int = 10, **kw):
        n = n_periods

        # (rolling close max - close) / (close - rolling close min)
        # NOTE could try slicing values at min 0

        kw['signals'] = dict(
            # ratio_high=lambda x: (x.close.rolling(n).max().shift(-n) - x.close) / x.close,
            # ratio_low=lambda x: (x.close.rolling(n).min().shift(-n) - x.close) / x.close,
            # target=lambda x: x.ratio_high + x.ratio_low,
            target=lambda x: 100 * (-2 + (x.close.rolling(n).max() + x.close.rolling(n).min()).shift(-n) / x.close)
        )

        super().__init__(n_periods=n_periods, **kw)


def add_emas(df, emas: list = None):
    """Convenience func to add both 50 and 200 emas"""
    if emas is None:
        emas = [50, 200]  # default fast/slow

    for p in emas:
        df = df.pipe(add_ema, p=p)

    return df


def add_ema(df: pd.DataFrame, p: int, c='close', col=None, overwrite=True):
    """Add ema from close price to df if column doesn't already exist (more than one signal may add an ema"""
    if col is None:
        col = f'ema{p}'

    if not col in df.columns or overwrite:
        # df[col] = df[c].ewm(span=p, min_periods=p).mean()
        df[col] = EMAIndicator(close=df[c], window=int(p), fillna=True).ema_indicator()

    return df


def get_mom_acc(s: pd.Series, size: int = 1):
    from findiff import FinDiff
    d_dx = FinDiff(0, size, 1)
    d2_dx2 = FinDiff(0, size, 2)
    arr = np.asarray(s)
    mom = d_dx(arr)
    momacc = d2_dx2(arr)
    return mom, momacc


def get_extrema(is_min, mom, momacc, s, window: int = 1):

    if not isinstance(s, list):
        s = s.tolist()

    return [x for x in range(len(mom))
            if (momacc[x] > 0 if is_min else momacc[x] < 0) and
            (mom[x] == 0 or  # slope is 0
            (x != len(mom) - 1 and  # check next day
                (mom[x] > 0 and mom[x + 1] < 0 and
                 s[x] >= s[x + 1] or
                 mom[x] < 0 and mom[x + 1] > 0 and
                 s[x] <= s[x + 1]) or
             x != 0 and  # check prior day
                (mom[x - 1] > 0 and mom[x] < 0 and
                 s[x - 1] < s[x] or
                 mom[x - 1] < 0 and mom[x] > 0 and
                 s[x - 1] > s[x])))]

    # momentum greater than zero, momentum next less than zero
    # AND price greater than next price
    #  OR
    # momentum less than zero and momentum next gt zero
    # ABD price less than next price
    #


def add_extrema(df, side):
    import peakutils  # type: ignore # not installed
    if side == 1:
        name = 'maxima'
        col = 'high'
        func = 'max'
        op = opr.gt
        is_min = False
    else:
        name = 'minima'
        col = 'low'
        func = 'min'
        op = opr.lt
        is_min = True

    s = df[col]
    if side == -1:
        s = 1 / s
    # mom, momacc = get_mom_acc(s=s, size=1)
    # df_mom = pd.DataFrame(data={f'mom_{func}': mom, f'mom_acc_{func}': momacc}, index=df.index)
    # idxs = get_extrema(is_min, mom, momacc, s)
    idxs = peakutils.indexes(s, thres=0.2, min_dist=24, thres_abs=False)
    df_extrema = df.iloc[idxs][col].rename(name)

    df = df.join(df_extrema) \
        .assign(
            **{
                f'rolling_{func}': lambda x: getattr(x[col].rolling(6), func)().shift(1),
                name: lambda x: np.where(op(x[name], x[f'rolling_{func}']), x[name], np.NaN),
                f'{name}_fwd': lambda x: x[name].ffill()}) \
        # .join(df_mom)

    return df


def _get_extrema(is_min, mom, momacc, h, window: int = 1):

    length = len(mom)
    op = opr.gt
    op = {True: opr.gt, False: opr.lt}.get(is_min)
    lst = []

    # for x in range(window, length - window):
    #     if op(momacc[x], 0) and (
    #         mom[x] == 0 or x != length - 1

    #     )

    return lst


def relative_self(s: pd.Series, n: int = 24) -> pd.Series:
    """Calculate column relative to its mean of previous n_periods"""
    return (s / np.abs(s).rolling(n).mean()).astype(np.float32)
