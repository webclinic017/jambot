import copy
import inspect
import operator as opr
import sys
from collections import defaultdict as dd

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, minmax_scale
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

from jambot import functions as f
from jambot import sklearn_utils as sk

from .__init__ import *

log = getlog(__name__)


class SignalManager():
    def __init__(
            self,
            signals_list: list = None,
            target: str = None,
            slope: int = 0,
            sum: int = 0,
            cut_periods: int = 200):

        signal_groups = {}
        features = {}  # map of {feature_name: signal_group}
        scaler = MinMaxScaler()
        f.set_self(vars())

    def fit(self, X, y=None, **fit_params):
        print('fit')
        return self

    def get_params(self, deep=True):
        # NOTE not using as a transformer yet
        return dict(thing1=1, thing2=2)

    def get_signal_group(self, feature_name: str) -> 'SignalGroup':
        """Return signal_group object from feature_name"""
        return list(filter(lambda x: x.has_feature(feature_name), self.signal_groups.values()))[0]

    def print_signal_params(self, **kw):
        m = self.get_signal_params(**kw)
        sk.pretty_dict(m)

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

    def add_signals(
            self,
            df: pd.DataFrame,
            signals: list = None,
            signal_params: dict = None,
            **kw) -> pd.DataFrame:
        """Add multiple initialized signals to dataframe"""
        # df = self.df
        if signals is None:
            signals = self.signals_list

        # convert input dict/single str to list
        if isinstance(signals, dict):
            signals = list(signals.values())

        signals = f.as_list(signals)
        signal_params = signal_params or {}
        final_signals = {}
        drop_cols = []

        # SignalGroup obj, not single column
        for signal_group in signals or []:

            # if str, init obj with defauls args
            if isinstance(signal_group, str):
                signal_group = getattr(sys.modules[__name__], signal_group)()

            self.signal_groups[signal_group.class_name] = signal_group

            signal_group.df = df  # NOTE kinda sketch
            signal_group.slope = self.slope  # sketch too
            signal_group.sum = self.sum
            # df = df.pipe(signal_group.add_all_signals, **kw)
            final_signals |= signal_group.make_signals(df=df, **kw)
            signal_group.df = None
            drop_cols += signal_group.drop_cols

        # saves ~70mb peak
        dtypes = {np.float32: np.float16}

        # .astype(np.float64) \
        # remove first rows that can't be set with 200ema accurately
        return df.assign(**final_signals) \
            .pipe(f.safe_drop, cols=drop_cols) \
            .pipe(f.reduce_dtypes, dtypes=dtypes) \
            .iloc[self.cut_periods:, :] \
            .fillna(0)

    def show_signals(self):
        m = dd(dict)
        for name, signal_group in self.signal_groups.items():

            for sig_name, params in signal_group.signals.items():
                if 'cls' in params:
                    m[name].update({sig_name: params['cls'].__name__})
                else:
                    m[name].update({sig_name: ''})

        sk.pretty_dict(m)

    def replace_single_feature(self, df, feature_params: dict, **kw) -> pd.DataFrame:
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


class SignalGroup():
    """Base class for ta indicators"""
    # Default OHLCV cols for df, used to pass to ta signals during init
    m_default = dict(
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume')

    def __init__(self, df=None, signals=None, fillna=True, prefix: str = None, **kw):
        drop_cols, no_slope_cols, no_sum_cols, normal_slope_cols = [], [], [], []
        signals = self.init_signals(signals)
        slope = 0
        sum = 0
        class_name = self.__class__.__name__.lower()
        f.set_self(vars())

    def init_signals(self, signals):
        """Extra default processing to signals, eg add plot func"""
        if signals is None:
            return

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
            .pipe(f.safe_drop, cols=self.drop_cols)

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

        final_signals = {}
        signals = self.signals if params is None else params

        # filter specific names to init specific signals only
        if not names is None:
            signals = {k: v for k, v in signals.items() if k in f.as_list(names)}

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
        exclude = self.drop_cols + self.no_slope_cols

        return {
            f'dyx_{c}': lambda x, c=c: self.make_slope(
                s=x[c],
                n_periods=self.slope,
                normal_slope=c in self.normal_slope_cols) for c in signals if not c in exclude}

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
        return {
            f'sum_{c}': lambda x, c=c:
                x[c].rolling(self.sum).sum().astype(np.float32) for c in signals if not c in exclude}

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
        ta_func = f.as_list(ta_func)

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
        if normal_slope:
            return (s.pct_change(n_periods) / n_periods).astype(np.float32)
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

        f.set_self(vars())

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
            rsi_2=dict(cls=RSIIndicator, ta_func='rsi', window=window, params=dict(window=[2, 6, 12, 18, 24, 36])),
            rsi_6=dict(cls=RSIIndicator, ta_func='rsi', window=6),
            rsi_12=dict(cls=RSIIndicator, ta_func='rsi', window=12),
            # rsi_24=dict(cls=RSIIndicator, ta_func='rsi', window=24),
            pvo=dict(cls=PercentageVolumeOscillator, ta_func='pvo'),
            roc=dict(cls=ROCIndicator, ta_func='roc', window=window),
            stoch=dict(cls=StochasticOscillator, ta_func='stoch', window=12, smooth_window=12),
            tsi=dict(cls=TSIIndicator, ta_func='tsi'),
            ultimate=dict(cls=UltimateOscillator, ta_func='ultimate_oscillator'),
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

        super().__init__(prefix='mnt', **kw)
        drop_cols = ['awesome']
        f.set_self(vars())


class Volume(SignalGroup):
    def __init__(self, **kw):
        prefix = 'vol'
        kw['signals'] = dict(
            # relative=lambda x: x.volume / x.volume.shift(6).rolling(24).mean(),
            relative=lambda x: relative_self(x.volume, n=24 * 7),
            # mom=lambda x: x.vol_relative * x.pct,
            chaik=dict(cls=ChaikinMoneyFlowIndicator, ta_func='chaikin_money_flow', window=4),
            mfi=dict(cls=MFIIndicator, ta_func='money_flow_index', window=48),
            # adi=dict(cls=AccDistIndexIndicator, ta_func='acc_dist_index'),
            # eom=dict(cls=EaseOfMovementIndicator, ta_func='ease_of_movement', window=14),
            # force=dict(cls=ForceIndexIndicator, ta_func='force_index', window=14),
        )

        super().__init__(**kw)
        f.set_self(vars())


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

        super().__init__(**kw)
        drop_cols = ['spread', 'vty_maxhigh', 'vty_minlow']
        f.set_self(vars())

    def final(self, c):
        # return self.df.norm_ema[i]
        return c.norm_ema


class Trend(SignalGroup):
    def __init__(self, **kw):
        kw['signals'] = dict(
            adx=dict(cls=ADXIndicator, ta_func='adx', window=36),
            aroon=dict(cls=AroonIndicator, ta_func='aroon_indicator', window=25),
            cci=dict(cls=CCIIndicator, ta_func='cci', window=20, constant=0.015),
            # mass=dict(cls=MassIndex, ta_func='mass_index', window_fast=9, window_slow=25),
            stc=dict(cls=STCIndicator, ta_func='stc'),
            # dpo=dict(cls=DPOIndicator, ta_func='dpo'),
            # kst=dict(cls=KSTIndicator, ta_func='kst'),
            # trix=dict(cls=TRIXIndicator, ta_func='trix', window=48),
            # psar=dict(cls=PSARIndicator, ta_func=['psar', 'psar_down', 'psar_up'],
            # # step=0.02, max_step=0.2, fillna=False)

        )

        super().__init__(prefix='trend', **kw)
        # accept 1 to n series of trend signals, eg 1 or -1
        f.set_self(vars())

    def final(self, c):
        return c.trend


class EMA(SignalGroup):
    """Add pxhigh, pxlow for rolling period, depending if ema_trend is positive or neg

    w = with, a = agains, n = neutral (neutral not used)
    """

    def __init__(self, fast=50, slow=200, speed=(24, 18), offset=1, **kw):

        against, wth, neutral = speed[0], speed[1], int(np.average(speed))
        colfast, colslow = f'ema_{fast}', f'ema_{slow}'
        c = self.get_c(maxspread=0.1)

        emas = [10, 50, 200]
        m_emas = {f'ema_{n}': dict(cls=EMAIndicator, ta_func='ema_indicator', window=n) for n in emas}
        kw['signals'] = copy.copy(m_emas)

        kw['signals'] |= dict(
            ema_spread=lambda x: (x[colfast] - x[colslow]) / ((x[colfast] + x[colslow]) / 2),
            # ema_trend=lambda x: np.where(x[colfast] > x[colslow], 1, -1),
            ema_conf=lambda x: self.ema_exp(x=x.ema_spread, c=c),
        )

        super().__init__(**kw)
        # drop_cols = ['mha', 'mhw', 'mla', 'mlw', 'mhn', 'mln']
        no_sum_cols = list(m_emas)
        normal_slope_cols = list(m_emas)
        no_slope_cols = ['ema_trend']
        f.set_self(vars())

    # def add_all_signals(self, df):
    #     return df \
    #         .pipe(add_emas, emas=[self.fast, self.slow]) \
    #         .pipe(super().add_all_signals)

    def final(self, side, c):
        temp_conf = abs(c.ema_conf)

        if side * c.ema_trend == 1:
            conf = 1.5 - temp_conf * 2
        else:
            conf = 0.5 + temp_conf * 2

        return conf * self.weight

    def get_c(self, maxspread):
        m = -2.9
        b = 0.135
        return round(m * maxspread + b, 2)

    def ema_exp(self, x, c):
        side = np.where(x >= 0, 1, -1)
        x = abs(x)

        aLim = 2
        a = -1000
        b = 3
        d = -3
        g = 1.7

        y = side * (a * x ** b + d * x ** g) / (aLim * a * x ** b + aLim * d * x ** g + c)

        return y.astype(np.float32)


class MACD(SignalGroup):
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
        f.set_self(vars())

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
        f.set_self(vars())

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
            .pipe(f.drop_cols, expr=r'^(high|low)_\d+') \
            .assign(**self.add_sum(signals=f.filter_cols(df, 'sfp')))


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
            cdl_side=lambda x: np.where(x.close > x.open, 1, -1).astype(np.int8),
            cdl_full=lambda x: (x.high - x.low) / x.open,
            cdl_body=lambda x: (x.close - x.open) / x.open,
            cdl_full_rel=lambda x: relative_self(x.cdl_full, n=n_periods),
            cdl_body_rel_6=lambda x: relative_self(x.cdl_body, n=6),
            cdl_body_rel_12=lambda x: relative_self(x.cdl_body, n=12),
            cdl_body_rel_24=lambda x: relative_self(x.cdl_body, n=24),
            cdl_tail_high=lambda x: np.abs(x.high - x[['close', 'open']].max(axis=1)) / x.open,
            cdl_tail_low=lambda x: np.abs(x.low - x[['close', 'open']].min(axis=1)) / x.open,
            ema200_v_high=lambda x: np.abs(x.high - x.ema_200) / x.open,
            ema200_v_low=lambda x: np.abs(x.low - x.ema_200) / x.open,
            pxhigh=lambda x: x.high.rolling(n_periods).max().shift(1),
            pxlow=lambda x: x.low.rolling(n_periods).min().shift(1),
            high_above_prevhigh=lambda x: np.where(x.high > x.pxhigh, 1, 0).astype(bool),
            close_above_prevhigh=lambda x: np.where(x.close > x.pxhigh, 1, 0).astype(bool),
            low_below_prevlow=lambda x: np.where(x.low < x.pxlow, 1, 0).astype(bool),
            close_below_prevlow=lambda x: np.where(x.close < x.pxlow, 1, 0).astype(bool),
            # buy_pressure=lambda x: (x.close - x.low.rolling(2).min().shift(1)) / x.close,
            # sell_pressure=lambda x: (x.close - x.high.rolling(2).max().shift(1)) / x.close,
        )

        # close v range
        ns = (6, 12, 24, 48, 96, 192)
        m_cvr = {f'cvr_{n}': lambda x, n=n: self.close_v_range(x, n_periods=n) for n in ns}

        kw['signals'] |= m_cvr

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
            'pxlow']

        f.set_self(vars())

    def close_v_range(self, df, n_periods=24) -> pd.Series:
        # col = f'close_v_range_{n_periods}'
        # dont really need to use .shift(1) here
        min_n = df.low.rolling(n_periods).min()
        range_n = (df.high.rolling(n_periods).max() - min_n)
        s = (df.close - min_n) / range_n
        s = s.fillna(s.mean())
        return s.astype(np.float32)

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
        f.set_self(vars())


class TargetClass(SignalGroup):
    """
    target classification
    - 3 classes
    - higher, lower, or same in x periods into future (10?)
    - same = (within x%, 0.5? > scale this based on daily volatility)
    - calc close price vs current price, x periods in future (rolling method)
    # NOTE doesn't work currently need to redo with kw['signals]
    """

    def __init__(self, p_ema=10, n_periods=10, pct_min=0.02, **kw):
        super().__init__(**kw)
        ema_col = f'ema_{p_ema}'  # named so can drop later
        pct_min = pct_min / 2

        f.set_self(vars())


class TargetMeanEMA(TargetClass):
    def __init__(self, **kw):
        super().__init__(**kw)
        f.set_self(vars())

    def add_all_signals(self, df):
        pct_min = self.pct_min  # NOTE this could be a hyperparam
        # TODO ^ definitely needs to be scaled to daily volatility

        predict_col = self.ema_col
        # predict_col = 'close'

        return df \
            .pipe(add_ema, p=self.p_ema) \
            .assign(
                pct_future=lambda x: (x[predict_col].shift(-1 * self.n_periods) - x[predict_col]) / x[predict_col],
                target=lambda x: np.where(x.pct_future > pct_min, 1, np.where(x.pct_future < pct_min * -1, -1, 0))) \
            .drop(columns=['pct_future'])


class TargetMean(TargetClass):
    """
    Calc avg price of next n close prices
    - maybe - bin into % ranges?
    """

    def __init__(self, n_periods, regression=False, pct_min=0, **kw):

        if not regression:
            kw['signals'] = dict(
                pct_future=lambda x: (x.close.shift(-1 * n_periods).rolling(n_periods).mean() - x.close),
                target=lambda x: np.where(x.pct_future > pct_min, 1, -1).astype(np.int8),
            )
            drop_cols = ['pct_future']

        else:
            # just use % diff for regression instead of -1/1 classification
            kw['signals'] = dict(
                target=lambda x: (x.close.shift(-1 * n_periods).rolling(n_periods).mean() - x.close) / x.close)

        super().__init__(**kw)

        f.set_self(vars())


class TargetMaxMin(TargetClass):
    """
    Create two outputs - min_low, max_high
    """

    def __init__(self, n_periods: int, **kw):

        items = [('max', 'high'), ('min', 'low')]
        kw['signals'] = {f'target_{fn}': lambda x, fn=fn, c=c: (
            x[c]
            .rolling(n_periods).__getattribute__(fn)()
            .shift(-n_periods) - x.close) / x.close for fn, c in items}

        super().__init__(**kw)

        f.set_self(vars())


class TargetUpsideDownside(TargetMaxMin):
    """
    - NOTE last n_periods of target will be -1, NOT NaN > need to make sure to drop
    """

    def __init__(self, n_periods: int, **kw):
        super().__init__(n_periods=n_periods, **kw)

        self.signals |= dict(
            target=lambda x: np.where(np.abs(x.target_max) > np.abs(x.target_min), 1, -1).astype(np.int8)
        )

        drop_cols = ['target_max', 'target_min']

        self.signals = self.init_signals(self.signals)

        f.set_self(vars())


class WeightedPercent(TargetMaxMin):
    """
    Create array of normalized weights based on absolute movement up/down from current close in next n periods
    - eg weight periods where lots of movement happens higher = more consequential
    - Also optional weight based on age eg np.linspace 0.5 > 1.0
    """

    def __init__(self, n_periods: int, **kw):
        super().__init__(n_periods=n_periods, **kw)

        drop_cols = ['target_max', 'target_min']

        self.signals |= dict(
            weight=lambda x: minmax_scale(
                x[drop_cols].abs().max(axis=1) * np.linspace(0.5, 1, len(x)),
                feature_range=(0, 1)))
        # pd.qcut(
        #     x=x[drop_cols].abs().max(axis=1),
        #     q=100,
        #     labels=False,
        #     duplicates='drop'),  # * np.linspace(0.5, 1, len(x))

        self.signals = self.init_signals(self.signals)
        f.set_self(vars())

    def get_weight(self, df: pd.DataFrame) -> pd.Series:
        """return single array of weighted values to pass as fit_params

        Parameters
        ----------
        df : pd.DataFrame
            df with ['high', 'low', 'close']

        Returns
        -------
        pd.Series
            column weighted by abs movement up/down in next n_periods
        """
        cols = ['high', 'low', 'close']
        return df[cols] \
            .pipe(self.add_all_signals) \
            .assign(weight=lambda x: x.weight.fillna(x.weight.mean()).astype(np.float32))['weight']


def add_emas(df, emas: list = None):
    """Convenience func to add both 50 and 200 emas"""
    if emas is None:
        emas = [50, 200]  # default fast/slow

    for p in emas:
        df = df.pipe(add_ema, p=p)

    return df


def add_ema(df, p, c='close', col=None, overwrite=True):
    """Add ema from close price to df if column doesn't already exist (more than one signal may add an ema"""
    if col is None:
        col = f'ema_{p}'

    if not col in df.columns or overwrite:
        # df[col] = df[c].ewm(span=p, min_periods=p).mean()
        df[col] = EMAIndicator(close=df[c], window=p, fillna=True).ema_indicator()

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
    import peakutils
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
