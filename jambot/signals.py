import sys

import numpy as np
import ta

from . import functions as f

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

drop
- Open, High, Low, Close > don't care about absolute values
- timestamp, just keep as index
- drop anything still NA at beginning of series
"""


def add_signals(df, signals=None):
    """Add multiple initialized signals to dataframe"""
    if isinstance(signals, dict): signals = signals.values()

    for signal in signals or []:

        # if str, init obj with defauls args
        if isinstance(signal, str):
            signal = getattr(sys.modules[__name__], signal)()

        df = df.pipe(signal.add_signal)

    return df

class Signal():
    def __init__(self, weight=1):
        trendseries = None
        f.set_self(vars())
    
    def trend_changed(self, df, i, side):
        tseries = df[self.trendseries]
        tnow = tseries.iloc[i]
        tprev = tseries.iloc[i - 1]
        return not tnow == side and not tnow == tprev

class MACD(Signal):
    def __init__(self, fast=50, slow=200, smooth=50, **kw):
        super().__init__(**kw)
        name = 'macd'
        trendseries = 'macd_trend'
        f.set_self(vars())
    
    def add_signal(self, df):
        fast, slow, smooth = self.fast, self.slow, self.smooth

        return df \
            .pipe(add_ema, p=fast) \
            .pipe(add_ema, p=slow) \
            .assign(
                macd=lambda x: x[f'ema{fast}'] - x[f'ema{slow}'],
                macd_signal=lambda x: x.macd.ewm(span=smooth, min_periods=smooth).mean(),
                macd_diff=lambda x: x.macd - x.macd_signal,
                macd_trend=lambda x: np.where(x.macd_diff > 0, 1, -1))

    def final(self, side, c):
        conf = 1.25 if side * c.macd_trend == 1 else 0.5
        return conf * self.weight        

class RSI(Signal):
    def __init__(self, window=6, **kw):
        super().__init__(**kw)
        name = 'rsi'
        f.set_self(vars())
    
    def add_signal(self, df):
        rsi = ta.momentum.RSIIndicator(close=df.Close, window=self.window, fillna=True)
        # rsi_stoch = ta.momentum.StochRSIIndicator(close=df.Close, window=self.window, smooth1=8, smooth2=8, fillna=True)

        return df \
            .assign(
                rsi=rsi.rsi(),
                # rsi_stoch=rsi_stoch.stochrsi(),
                # rsi_stoch_k=rsi_stoch.stochrsi_k(),
                # rsi_stoch_d=rsi_stoch.stochrsi_d()
                )

class EMA(Signal):
    def __init__(self, fast=50, slow=200, **kw):
        super().__init__(**kw)
        name = 'ema'
        colfast, colslow = f'ema{fast}', f'ema{slow}'
        trandseries = 'ema_trend'
        f.set_self(vars())
    
    def add_signal(self, df):
        colfast, colslow = self.colfast, self.colslow
        c = self.get_c(maxspread=0.1)

        return df \
            .pipe(add_ema, p=self.fast) \
            .pipe(add_ema, p=self.slow) \
            .assign(
                ema_spread=lambda x: (x[colfast] - x[colslow]) / ((x[colfast] + x[colslow]) / 2),
                ema_trend=lambda x: np.where(x[colfast] > x[colslow], 1, -1),
                ema_conf=lambda x: self.ema_exp(x=x.ema_spread, c=c))

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

class EMASlope(Signal):
    """Calc slope (+1 or -1) for ema, over previous slope periods"""
    def __init__(self, p=50, slope=5, **kw):
        super().__init__(**kw)
        name = 'ema_Slope'
        trendseries = 'ema_slope'
        f.set_self(vars())
    
    def add_signal(self, df):
        p, slope = self.p, self.slope
        ema_col = f'ema{p}'

        return df \
            .pipe(add_ema, p=p) \
            .assign(
                ema_slope=lambda x: np.where(np.roll(x[ema_col], slope, axis=0) < x[ema_col], 1, -1))
        df.loc[:p + slope, 'ema_slope'] = np.nan

    def final(self, side, c):
        conf = 1.5 if side * c.ema_slope == 1 else 0.5
        return conf * self.weight     

class Volatility(Signal):
    def __init__(self, norm=(0.004, 0.024), **kw):
        super().__init__(**kw)
        name = 'volatility'
        f.set_self(vars())
    
    def add_signal(self, df):
        norm = self.norm

        return df \
            .assign(
                maxhigh=lambda x: x.High.rolling(48).max(),
                minlow=lambda x: x.Low.rolling(48).min(),
                vty_spread=lambda x: abs(x.maxhigh - x.minlow) / x[['maxhigh', 'minlow']].mean(axis=1),
                vty_ema=lambda x: x.vty_spread.ewm(span=60, min_periods=60).mean(),
                vty_sma=lambda x: x.vty_spread.rolling(300).mean(),
                norm_ema=lambda x: np.interp(x.vty_ema, (0, 0.25), (norm[0], norm[1])),
                norm_sma=lambda x: np.interp(x.vty_sma, (0, 0.25), (norm[0], norm[1]))) \
            .drop(columns=['maxhigh', 'minlow'])

    def final(self, c):
        # return self.df.norm_ema[i]
        return c.norm_ema

class Trend(Signal):
    """Add pxhigh, pxlow for rolling period, depending if ema_trend is positive or neg
    
    w = with, a = agains, n = neutral (neutral not used)
    """
    def __init__(self, signal_series='ema_trend', speed=None, offset=1, **kw):
        super().__init__(**kw)
        # accept 1 to n series of trend signals, eg 1 or -1
        if speed is None:
            speed = (24, 18) # against/with

        f.set_self(vars())
    
    def add_signal(self, df):
        speed, offset = self.speed, self.offset
        against, wth, neutral = speed[0], speed[1], int(np.average(speed))

        # set trade high/low in period prices
        return df \
            .assign(
                trend=lambda x: x[self.signal_series],
                mhw=lambda x: x.High.rolling(wth).max().shift(offset),
                mha=lambda x: x.High.rolling(against).max().shift(offset),
                mhn=lambda x: x.High.rolling(neutral).max().shift(offset),
                mla=lambda x: x.Low.rolling(wth).min().shift(offset),
                mlw=lambda x: x.Low.rolling(against).min().shift(offset),
                mln=lambda x: x.Low.rolling(neutral).min().shift(offset),
                pxhigh=lambda x: np.where(x.trend == 0, x.mhn, np.where(x.trend == 1, x.mha, x.mhw)),
                pxlow=lambda x: np.where(x.trend == 0, x.mln, np.where(x.trend == -1, x.mlw, x.mla)),) \
            .drop(columns=['mha', 'mhw', 'mla', 'mlw', 'mhn', 'mln'])
        
    def final(self, c):
        return c.trend

class SFP(Signal):
    """
    Calculate wether candle is sfp, for any of 3 previous max/min periods
    - This signal needs cdl body signals init first
    - NOTE could maybe use min/max peaks from tsfresh?
    """
    def __init__(self, period_base=48, **kw):
        super().__init__(**kw)
        minswing = 0.05 # NOTE could be hyperparam
        f.set_self(vars())

    def add_signal(self, df):

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

class Candle(Signal):
    """
    - cdl_side > wether candle is red or green > (open - close)
    - cdl_size_full > size of total candle (% high - low) > relative to volatility
    - cdl_size_body > size of candle body (% close - open)
    - tail_size_low, tail_size_high > size of tails (% close - low, high - low) > scale to volatility?
    - 200_v_high, 200_v_low > % proximity of candle low/high to 200ema
        - close to support with large tail could mean reversal
    - TODO high_above_prevhigh, close_above_prevhigh > maybe add in trend? similar to close_v_range
    - TODO low_below_prevlow, close_below_prevlow
    - close_v_range > did we close towards upper or lower side of prev 24 period min/max range
    """
    def __init__(self, **kw):
        super().__init__(**kw)

    def add_signal(self, df):
        # NOTE could be kinda wrong div by Open, possibly try div by SMA?
        n_periods = 24 # used to cal relative position of close to prev range
        # TODO NEED candle body sizes relative to current rolling volatility

        return df \
            .pipe(add_both_emas) \
            .assign(
                cdl_side=lambda x: np.where(x.Close > x.Open, 1, -1),
                cdl_size_full=lambda x: np.abs(x.High - x.Low) / x.Open,
                cdl_size_body=lambda x: np.abs(x.Close - x.Open) / x.Open,
                cdl_tail_size_high=lambda x: np.abs(x.High - x[['Close', 'Open']].max(axis=1)) / x.Open,
                cdl_tail_size_low=lambda x: np.abs(x.Low - x[['Close', 'Open']].min(axis=1)) / x.Open,
                ema200_v_high=lambda x: np.abs(x.High - x.ema200) / x.Open,
                ema200_v_low=lambda x: np.abs(x.Low - x.ema200) / x.Open,
                min_n=lambda x: x.Low.rolling(n_periods).min(),
                range_n=lambda x: (x.High.rolling(n_periods).max() - x.min_n),
                close_v_range=lambda x: (x.Close - x.min_n) / x.range_n,
                high_above_prevhigh=lambda x: np.where(x.High > x.pxhigh, 1, 0),
                close_above_prevhigh=lambda x: np.where(x.Close > x.pxhigh, 1, 0),
                low_below_prevlow=lambda x: np.where(x.Low < x.pxlow, 1, 0),
                close_below_prevlow=lambda x: np.where(x.Close < x.pxlow, 1, 0),
            ) \
            .drop(columns=['min_n', 'range_n'])

class TargetClass(Signal):
    """
    target classification
    - 3 classes
    - higher, lower, or same in x periods into future (10?)
    - same = (within x%, 0.5? > scale this based on daily volatility)
    - calc close price vs current price, x periods in future (rolling method)
    """
    def __init__(self, p_ema=10, n_periods=10, pct_min=0.02, **kw):
        super().__init__(**kw)
        ema_col = f'ema{p_ema}' # named so can drop later
        pct_min = pct_min / 2

        f.set_self(vars())
    
    def add_signal(self, df):
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

def add_both_emas(df):
    """Convenience func to add both 50 and 200 emas"""
    return df \
        .pipe(add_ema, p=50) \
        .pipe(add_ema, p=200)

def add_ema(df, p, c='Close'):
    """Add ema from Close price to df if column doesn't already exist (more than one signal may add an ema"""
    col = f'ema{p}'
    if not col in df.columns:
        df[col] = df[c].ewm(span=p, min_periods=p).mean()
    
    return df
