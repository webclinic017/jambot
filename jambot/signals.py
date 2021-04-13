import inspect
import re
import sys
from collections import defaultdict as dd
import operator as opr

import numpy as np
import pandas as pd
import peakutils
import ta
import talib as tb
from findiff import FinDiff
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import (AwesomeOscillatorIndicator, KAMAIndicator,
                         PercentageVolumeOscillator, ROCIndicator,
                         RSIIndicator, StochasticOscillator, StochRSIIndicator,
                         TSIIndicator, UltimateOscillator)
from ta.trend import (ADXIndicator, AroonIndicator, CCIIndicator, DPOIndicator,
                      EMAIndicator, KSTIndicator, MassIndex, STCIndicator,
                      TRIXIndicator)
from ta.volatility import UlcerIndex
from ta.volume import (AccDistIndexIndicator, ChaikinMoneyFlowIndicator,
                       EaseOfMovementIndicator, ForceIndexIndicator,
                       MFIIndicator)

from . import charts as ch
from . import functions as f
from . import sklearn_helper_funcs as sf

"""
target classification
- 3 classes
- higher, lower, or same in x periods into future (10?)
- same = (within x%, 0.5? > scale this based on daily volatility)
- calc close price vs current price, x periods in future (rolling method)

FEATURES
- CANDLE
    - cdl_size_full > size of total candle (% high - low) > relative to volatility
    - cdl_size_body > size of candle body (% close - open)
    - tail_size_low, tail_size_high > size of tails (% close - low, high - low) > scale to volatility?
    - 200_v_high, 200_v_low > % proximity of candle low/high to 200ema
        - close to support with large tail could mean reversal
    - cdl_type > wether candle is red or green > (open - close)
    - high_above_prevhigh, close_above_prevhigh
    - low_below_prevlow, close_below_prevlow
    - close vs mean in x periods, eg did we close at higher or lower end of current 'range'

- SWING FAIL (6 cols)
    - check 3 sfp with respect to prev rolling periods > 48, 96, 192
    - sfp_low_48, spf_high_192 etc

- EMA
    - ema_slope > 50ema positive or neg
    - ema_spread > % separation btwn ema50 and ema200
    - ema_trend > wether 50ema is above 200 or not
    - ema_conf > try it out, cant remember exactly
- VOLATILITY
    - vty_ema > ema of volatility (60 period rolling). should this still be normalized?
- MACD
    - macd_trend
- RSI
    - rsi (create from tech indicators or manually)
    - stoch rsi.. not sure


Potential ideas
- low/high proximity to prev spikes low? > calc spikes with tsfresh?
- number of prev spikes low, eg higher # of spikes up or down might mean support is starting to fail
- remove all higher/lower indicators, only classify wether trend will continue or reverse?
    - this simplifies everything, could lead to better convergence
    - long term up/down trend may bias results otherwise, would be bad if long term trend switched
- stacked entry > buy 1/3 every x periods > would probaly work better with trend

ta lib


drop
- Open, High, Low, Close > don't care about absolute values
- timestamp, just keep as index
- drop anything still NA at beginning of series
"""

class SignalManager(BaseEstimator, TransformerMixin):
    def __init__(self, signals_list : list=None):
        # df_orig = df.copy()
        signal_groups = {}
        features = {} # map of {feature_name: signal_group}
        scaler = MinMaxScaler()
        f.set_self(vars())
    
    def transform(self, df, **transform_params):
        """Return compound sentiment score for text column"""
        # self.df = df
        print(f'transforming: {df.shape}, {df.columns}')
        

        df = self.add_signals(df=df, signals=self.signals_list)
        cols_ohlcv = ['Open', 'High', 'Low', 'Close', 'VolBTC']
        drop_feats = [col for col in cols_ohlcv + ['ema10', 'ema50', 'ema200', 'pxhigh', 'pxlow'] if col in df.columns]

        df = df.drop(columns=drop_feats)
        self.cols = df.columns.to_list()

        return self.scaler.fit_transform(df)

    def fit(self, X, y=None, **fit_params):
        print('fit')
        return self

    def get_params(self, deep=True):
        # NOTE not using as a transformer yet
        return dict(thing1=1, thing2=2)
    
    def get_signal_group(self, feature_name : str):
        """Return signal_group object from feature_name"""
        return list(filter(lambda x: feature_name in x.signals, self.signal_groups.values()))[0]

    def print_signal_params(self, **kw):
        m = self.get_signal_params(**kw)
        sf.pretty_dict(m)
    
    def get_signal_params(self, feature_names=None, **kw):
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
        print('get_feature_names')
        return self.cols
        # return self.df.columns.to_list()

    def add_signals(self, df, signals : list=None, signal_params : dict=None) -> pd.DataFrame:
        """Add multiple initialized signals to dataframe"""
        # df = self.df

        # convert input dict/single str to list
        if isinstance(signals, dict): signals = list(signals.values())
        if isinstance(signals, str): signals = [signals]
        signal_params = signal_params or {}

        # SignalGroup obj, not single column
        for signal_group in signals or []:

            # if str, init obj with defauls args
            if isinstance(signal_group, str):
                signal_group = getattr(sys.modules[__name__], signal_group)()
            
            self.signal_groups[signal_group.__class__.__name__.lower()] = signal_group

            signal_group.df = df # NOTE kinda sketch
            df = df.pipe(signal_group.add_all_signals)

        self.df = df

        return df

    def replace_single_feature(self, df, feature_params : dict, **kw) -> pd.DataFrame:
        """Generator to replace single feature at a time with new values from params dict"""

        for feature_name, m in feature_params.items():
            # get signal group from feature_name
            signal_group = self.get_signal_group(feature_name=feature_name)

            for param, vals in m.items():              
                for val in vals:
                    param_single = {feature_name: {param: val}} # eg {'mnt_rsi': {'window': 12}}
                    yield df.pipe(signal_group.assign_signals, params=param_single), f'{feature_name}_{param}_{val}'

    
    def make_plot_traces(self, name) -> list:
        """Create list of dicts for plot traces"""
        name = name.lower()
        signal_group = self.signal_groups[name].signals

        return [dict(name=name, func=m['plot'], row=m.get('row', None)) for name, m in signal_group.items()]
    
    def plot_signal_group(self, name, **kw):
        """Convenience func to show plot with a full group of signals"""
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
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='VolBTC')

    def __init__(self, df=None, signals=None, fillna=True, **kw):
        drop_cols = []
        signals = self.init_signals(signals)
        f.set_self(vars())

    def init_signals(self, signals):
        """Extra default processing to signals, eg add plot func"""
        if signals is None:
            return

        for name, m in signals.items():
            
            if not isinstance(m, dict):
                if callable(m):
                    m = dict(func=m) # signal was just created as single func
                else:
                    raise AttributeError('signal must be init with a dict or callable func!')

            m['name'] = name

            # assign scatter as default plot
            if m.get('plot', None) is None:
                m['plot'] = ch.scatter

            signals[name] = m

        return signals
    
    def add_all_signals(self, df):
        """Convenience base wrapper to work with ta/non-ta signals"""
        df = df.pipe(self.assign_signals)

        drop_cols = [col for col in self.drop_cols if col in df.columns]

        return df \
            .drop(columns=drop_cols)
    
    def assign_signals(self, df, params : dict=None) -> pd.DataFrame:
        """loop self.signal_groups, init, call correct func and assign to df column
        
        Parameters
        ----------
        params : dict
            if params passed, only reassign those features (for quick optimization testing)
            params must be dict of {feature_name: {single param: value}}
        """

        final_signals = {}
        signals = self.signals if params is None else params

        for feature_name, m in signals.items():
            
            if not params is None:
                # overwrite original feature params with new value eg {'mnt_rsi': {'window': 12}}
                m = {**self.signals[feature_name], **m}

            if 'cls' in m:
                # init ta signal
                final_signals[feature_name] = self.make_signal(**m)
            elif 'func' in m:
                final_signals[feature_name] = m['func']
            else:
                raise AttributeError('No func to assign from signal!')

        return df \
            .assign(**final_signals)
    
    @property
    def default_cols(self) -> dict:
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
    
    def make_signal(self, cls, ta_func : str, **kw):
        """Helper func to init TA signal with correct OHLCV columns
        
        Parameters
        ---------
        cls : ta
            class defn of ta indicator to be init
        ta_func : str
            func to call on ta obj to create final signal
        """
        kw['fillna'] = self.fillna
        good_kw = self.filter_valid_kws(cls=cls, **kw)

        return getattr(cls(**good_kw), ta_func)()

    def trend_changed(self, df, i, side):
        """NOTE Not sure if this used"""
        tseries = df[self.trendseries]
        tnow = tseries.iloc[i]
        tprev = tseries.iloc[i - 1]
        return not tnow == side and not tnow == tprev

class Momentum(SignalGroup):
    def __init__(self, window=2, **kw):
        kw['signals'] = dict(
            mnt_rsi_2=dict(cls=RSIIndicator, ta_func='rsi', window=window, params=dict(window=[2, 6, 12, 18, 24, 36])),
            mnt_rsi_6=dict(cls=RSIIndicator, ta_func='rsi', window=6),
            mnt_rsi_24=dict(cls=RSIIndicator, ta_func='rsi', window=24),
            mnt_pvo=dict(cls=PercentageVolumeOscillator, ta_func='pvo'),
            mnt_roc=dict(cls=ROCIndicator, ta_func='roc', window=window),
            mnt_stoch=dict(cls=StochasticOscillator, ta_func='stoch', window=12, smooth_window=12),
            mnt_tsi=dict(cls=TSIIndicator, ta_func='tsi'),
            mnt_ultimate=dict(cls=UltimateOscillator, ta_func='ultimate_oscillator'),
            mnt_awesome=dict(cls=AwesomeOscillatorIndicator, ta_func='awesome_oscillator', window1=10, window2=50, params=dict(window1=[6, 12, 18], window2=[36, 50, 200])),
            mnt_awesome_rel=lambda x: x.mnt_awesome / x.ema50
            # mnt_kama=dict(cls=KAMAIndicator, ta_func='kama', window=12, pow1=2, pow2=30, row=1)
            )

        super().__init__(**kw)
        drop_cols = ['mnt_awesome']
        f.set_self(vars())

class Volume(SignalGroup):
    def __init__(self, **kw):
        kw['signals'] = dict(
            vol_relative=lambda x: x.VolBTC / x.VolBTC.shift(6).rolling(24).mean(),
            vol_chaik=dict(cls=ChaikinMoneyFlowIndicator, ta_func='chaikin_money_flow'),
            vol_mfi=dict(cls=MFIIndicator, ta_func='money_flow_index', window=14),
            # vol_adi=dict(cls=AccDistIndexIndicator, ta_func='acc_dist_index'),
            # vol_eom=dict(cls=EaseOfMovementIndicator, ta_func='ease_of_movement', window=14),
            # vol_force=dict(cls=ForceIndexIndicator, ta_func='force_index', window=14),
            )

        super().__init__(**kw)
        f.set_self(vars())
   
class Volatility(SignalGroup):
    def __init__(self, norm=(0.004, 0.024), **kw):
        kw['signals'] = dict(
            vty_ulcer=dict(cls=UlcerIndex, ta_func='ulcer_index', window=6),
            # maxhigh=lambda x: x.High.rolling(48).max(),
            # minlow=lambda x: x.Low.rolling(48).min(),
            # vty_spread=lambda x: abs(x.maxhigh - x.minlow) / x[['maxhigh', 'minlow']].mean(axis=1),
            # vty_ema=lambda x: x.vty_spread.ewm(span=60, min_periods=60).mean(),
            # vty_sma=lambda x: x.vty_spread.rolling(300).mean(),
            # norm_ema=lambda x: np.interp(x.vty_ema, (0, 0.25), (norm[0], norm[1])),
            # norm_sma=lambda x: np.interp(x.vty_sma, (0, 0.25), (norm[0], norm[1])),
        )

        super().__init__(**kw)
        drop_cols = ['maxhigh', 'minlow']
        f.set_self(vars())  

    def final(self, c):
        # return self.df.norm_ema[i]
        return c.norm_ema

class Trend(SignalGroup):
    def __init__(self, **kw):
        kw['signals'] = dict(
                trend_adx=dict(cls=ADXIndicator, ta_func='adx', window=36),
                trend_aroon=dict(cls=AroonIndicator, ta_func='aroon_indicator', window=25),
                trend_cci=dict(cls=CCIIndicator, ta_func='cci', window=20, constant=0.015),
                trend_mass=dict(cls=MassIndex, ta_func='mass_index', window_fast=9, window_slow=25),
                trend_stc=dict(cls=STCIndicator, ta_func='stc'),
                # trend_dpo=dict(cls=DPOIndicator, ta_func='dpo'),
                trend_kst=dict(cls=KSTIndicator, ta_func='kst'),
                trend_trix=dict(cls=TRIXIndicator, ta_func='trix', window=48),
                
        )
        
        super().__init__(**kw)
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
        colfast, colslow = f'ema{fast}', f'ema{slow}'
        c = self.get_c(maxspread=0.1)

        kw['signals'] = dict(
                ema_spread=lambda x: (x[colfast] - x[colslow]) / ((x[colfast] + x[colslow]) / 2),
                ema_trend=lambda x: np.where(x[colfast] > x[colslow], 1, -1),
                ema_conf=lambda x: self.ema_exp(x=x.ema_spread, c=c),
                mhw=lambda x: x.High.rolling(wth).max().shift(offset),
                mha=lambda x: x.High.rolling(against).max().shift(offset),
                mhn=lambda x: x.High.rolling(neutral).max().shift(offset),
                mla=lambda x: x.Low.rolling(wth).min().shift(offset),
                mlw=lambda x: x.Low.rolling(against).min().shift(offset),
                mln=lambda x: x.Low.rolling(neutral).min().shift(offset),
                pxhigh=dict(row=1, func=lambda x: np.where(x.ema_trend == 0, x.mhn, np.where(x.ema_trend == 1, x.mha, x.mhw))),
                pxlow=dict(row=1, func=lambda x: np.where(x.ema_trend == 0, x.mln, np.where(x.ema_trend == -1, x.mlw, x.mla))),
        )

        super().__init__(**kw)
        drop_cols = ['mha', 'mhw', 'mla', 'mlw', 'mhn', 'mln']
        # trandseries = 'ema_trend'
        f.set_self(vars())
    
    def add_all_signals(self, df):
        return df \
            .pipe(add_emas, emas=[self.fast, self.slow]) \
            .pipe(super().add_all_signals)

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

        return round(y, 6)

class EMASlope(SignalGroup):
    """Calc slope (+1 or -1) for ema, over previous slope periods"""
    def __init__(self, p=50, slope=5, **kw):
        ema_col = f'ema{p}'
        kw['signals'] = dict(
            ema10_slope=lambda x: (x.ema10 - np.roll(x.ema10, slope, axis=0)) / slope,
            ema50_slope=lambda x: (x.ema50 - np.roll(x.ema50, slope, axis=0)) / slope,
            ema50_slope_int=lambda x: np.where(np.roll(x[ema_col], slope, axis=0) < x[ema_col], 1, -1)
            )

        super().__init__(**kw)
        name = 'ema_Slope'
        trendseries = 'ema_slope'
        f.set_self(vars())
    
    def add_all_signals(self, df):
        p, slope = self.p, self.slope

        return df \
            .pipe(add_emas, emas=[10, 50]) \
            .pipe(super().add_all_signals)

        # df.loc[:p + slope, 'ema_slope'] = np.nan

    def final(self, side, c):
        conf = 1.5 if side * c.ema_slope == 1 else 0.5
        return conf * self.weight     

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
        minswing = 0.05 # NOTE could be hyperparam
        f.set_self(vars())

    def add_all_signals(self, df):

        # set min low/max high for each rolling period
        offset = 6
        periods = [self.period_base * 2 ** i for i in range(3)]

        m = dict(High=['max', 1], Low=['min', -1])

        for period in periods:
            for extrema, v in m.items():
                agg_func, side = v[0], v[1] 
                check_extrema = f'{extrema.lower()}_{period}' # high_48
                
                s = df[extrema].rolling(period)
                
                df[check_extrema] = getattr(s, agg_func)().shift(offset) # min() or max()
                # df[f'low_{period}'] = df.Low.rolling(period).min().shift(offset)

                # calc candle is swing fail or not
                # minswing means tail must also be greater than min % of candle full size
                df[f'sfp_{check_extrema}'] = np.where(
                    (side * (df[extrema] - df[check_extrema]) > 0) & 
                    (side * (df.Close - df[check_extrema] < 0)) &
                    (df[f'cdl_tail_size_{extrema.lower()}'] / df.cdl_size_full > self.minswing),
                    1, 0)

        drop_cols = [col for col in df.columns if any(item in col for item in ('high_', 'low_')) and not any(item in col for item in ('sfp', 'prevhigh', 'prevlow'))]

        return df \
            .drop(columns=drop_cols)

class Candle(SignalGroup):
    """
    - cdl_side > wether candle is red or green > (open - close)
    - cdl_size_full > size of total candle (% high - low) > relative to volatility
    - cdl_size_body > size of candle body (% close - open)
    - tail_size_low, tail_size_high > size of tails (% close - low, high - low) > scale to volatility?
    - 200_v_high, 200_v_low > % proximity of candle low/high to 200ema
        - close to support with large tail could mean reversal
    - high_above_prevhigh, close_above_prevhigh > maybe add in trend? similar to close_v_range
    - low_below_prevlow, close_below_prevlow
    - close_v_range > did we close towards upper or lower side of prev 24 period min/max range
    """
    def __init__(self, **kw):

        n_periods = 24 # used to cal relative position of close to prev range
        kw['signals'] = dict(
            cdl_side=lambda x: np.where(x.Close > x.Open, 1, -1),
            cdl_size_full=lambda x: np.abs(x.High - x.Low) / x.Open,
            cdl_size_body=lambda x: np.abs(x.Close - x.Open) / x.Open,
            cdl_tail_size_high=lambda x: np.abs(x.High - x[['Close', 'Open']].max(axis=1)) / x.Open,
            cdl_tail_size_low=lambda x: np.abs(x.Low - x[['Close', 'Open']].min(axis=1)) / x.Open,
            ema200_v_high=lambda x: np.abs(x.High - x.ema200) / x.Open,
            ema200_v_low=lambda x: np.abs(x.Low - x.ema200) / x.Open,
            # ema50_v_close=lambda x: (x.Close - x.ema50) / x.Close,
            # min_n=lambda x: x.Low.rolling(n_periods).min(),
            # range_n=lambda x: (x.High.rolling(n_periods).max() - x.min_n),
            # close_v_range=lambda x: (x.Close - x.min_n) / x.range_n,
            high_above_prevhigh=lambda x: np.where(x.High > x.pxhigh, 1, 0),
            close_above_prevhigh=lambda x: np.where(x.Close > x.pxhigh, 1, 0),
            low_below_prevlow=lambda x: np.where(x.Low < x.pxlow, 1, 0),
            close_below_prevlow=lambda x: np.where(x.Close < x.pxlow, 1, 0),
            buy_pressure=lambda x: (x.Close - x.Low.rolling(2).min().shift(1)) / x.Close,
            sell_pressure=lambda x: (x.Close - x.High.rolling(2).max().shift(1)) / x.Close,
        )


        super().__init__(**kw)
        drop_cols = ['min_n', 'range_n', 'pxhigh', 'pxlow']
        f.set_self(vars())

    def close_v_range(self, df, n_periods=24):
        col = f'close_v_range_{n_periods}'
        min_n = df.Low.rolling(n_periods).min()
        range_n = (df.High.rolling(n_periods).max() - min_n)
        df[col] = (df.Close - min_n) / range_n
        df[col] = df[col].fillna(df[col].mean())
        return df

    def add_all_signals(self, df):
        # NOTE could be kinda wrong div by Open, possibly try div by SMA?
        # TODO NEED candle body sizes relative to current rolling volatility

        return df \
            .pipe(add_emas) \
            .pipe(super().add_all_signals) \
            .pipe(lambda df: df.fillna(value=dict(
                buy_pressure=0,
                sell_pressure=0,
                # close_v_range=df.close_v_range.mean()
                ))) \
            .pipe(self.close_v_range, n_periods=96) \
            # .pipe(self.close_v_range, n_periods=24) \
            # .pipe(self.close_v_range, n_periods=384) \

class CandlePatterns(SignalGroup):
    def __init__(self, **kw):

        super().__init__()

    def add_all_signals(self, df):
        candle_names = tb.get_function_groups()['Pattern Recognition']

        for candle_name in candle_names:
            df[candle_name] = getattr(tb, candle_name)(df.Open, df.High, df.Low, df.Close)

        return df
        # return super().add_all_signals(df)

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
        ema_col = f'ema{p_ema}' # named so can drop later
        pct_min = pct_min / 2

        f.set_self(vars())

class TargetMeanEMA(TargetClass):
    def __init__(self, **kw):
        super().__init__(**kw)
        f.set_self(vars())
    
    def add_all_signals(self, df):
        pct_min = self.pct_min # NOTE this could be a hyperparam
        # TODO ^ definitely needs to be scaled to daily volatility

        predict_col = self.ema_col
        # predict_col = 'Close'

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
    def __init__(self, **kw):
        n_periods, pct_min = kw['n_periods'], kw['pct_min']

        kw['signals'] = dict(
            pct_future=lambda x: (x.Close.shift(-1 * n_periods).rolling(n_periods).mean() - x.Close),
            target=lambda x: np.where(x.pct_future > pct_min, 1, -1),
        )

        super().__init__(**kw)

        drop_cols = ['pct_future']
        f.set_self(vars())   


def add_emas(df, emas : list=None):
    """Convenience func to add both 50 and 200 emas"""
    if emas is None:
        emas = [50, 200] # default fast/slow

    for p in emas:
        df = df.pipe(add_ema, p=p)
    
    return df

def add_ema(df, p, c='Close'):
    """Add ema from Close price to df if column doesn't already exist (more than one signal may add an ema"""
    col = f'ema{p}'
    if not col in df.columns:
        # df[col] = df[c].ewm(span=p, min_periods=p).mean()
        df[col] = EMAIndicator(close=df.Close, window=p, fillna=True).ema_indicator()
    
    return df

def get_mom_acc(s: pd.Series, size: int=1):

    d_dx = FinDiff(0, size, 1)
    d2_dx2 = FinDiff(0, size, 2)
    arr = np.asarray(s)
    mom = d_dx(arr)
    momacc = d2_dx2(arr)
    return mom, momacc

def get_extrema(is_min, mom, momacc, s, window: int=1):

    if not isinstance(s, list):
        s = s.tolist()

    return [x for x in range(len(mom))
        if (momacc[x] > 0 if is_min else momacc[x] < 0) and
        (mom[x] == 0 or #slope is 0
            (x != len(mom) - 1 and #check next day
                (mom[x] > 0 and mom[x + 1] < 0 and
                s[x] >= s[x + 1] or
                mom[x] < 0 and mom[x + 1] > 0 and
                s[x] <= s[x + 1]) or
            x != 0 and #check prior day
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

    if side == 1:
        name = 'maxima'
        col = 'High'
        func = 'max'
        op = opr.gt
        is_min = False
    else:
        name = 'minima'
        col = 'Low'
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

def _get_extrema(is_min, mom, momacc, h, window: int=1):

    length = len(mom)
    op = opr.gt
    op = {True: opr.gt, False: opr.lt}.get(is_min)
    lst = []

    # for x in range(window, length - window):
    #     if op(momacc[x], 0) and (
    #         mom[x] == 0 or x != length - 1 


    #     )

    return lst

# find peaks
# is peak max of rolling previous n candles
# filter on momentum value too?
# check current price's proximity to
# weight x previous peaks based on momentum?

def test_peaks():
    periods = 500
    start = -1000
    df2 = df.iloc[start: start + periods] \
        .pipe(sg.add_extrema, side=1) \
        .pipe(sg.add_extrema, side=-1)
    
    df2

    traces = [
        dict(name='maxima', func=ch.trace_extrema, row=1),
        dict(name='minima', func=ch.trace_extrema, row=1),
        # dict(name='mom_max', func=ch.scatter, row=2),
        # dict(name='mom_acc_max', func=ch.scatter, row=2),
        # dict(name='mom_min', func=ch.scatter, row=3),
        # dict(name='mom_acc_min', func=ch.scatter, row=3)
        ]


    fig = ch.chart(df=df2, periods=periods, traces=traces, secondary_row_width=0.3)
    fig.show()