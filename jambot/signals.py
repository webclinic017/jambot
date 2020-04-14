import numpy as np

from . import (
    functions as f)


class Signal():
    def __init__(self, df, weight=1):
        self.df = df
        self.weight = weight
        self.trendseries = None
    
    def trend_changed(self, i, side):
        tseries = self.trendseries
        tnow = tseries.iloc[i]
        tprev = tseries.iloc[i - 1]
        return not tnow == side and not tnow == tprev

class MACD(Signal):
    def __init__(self, df, weight=1, fast=50, slow=200, smooth=50):
        super().__init__(df=df, weight=weight)
        # init macd series
        self.name = 'macd'
        f.add_ema(df=df, p=fast)
        f.add_ema(df=df, p=slow)

        df['macd'] = df[f'ema{fast}'] - df[f'ema{slow}']
        df['macd_signal'] = df.macd.ewm(span=smooth, min_periods=smooth).mean()
        df['macd_diff'] =  df.macd - df.macd_signal
        df['macd_trend'] = np.where(df.macd_diff > 0, 1, -1)

        self.trendseries = df.macd_trend

    def final(self, side, c):
        if side * c.macd_trend == 1:
            conf = 1.25
        else:
            conf = 0.5

        return conf * self.weight        

class EMA(Signal):
    def __init__(self, df, weight=1, fast=50, slow=200):
        super().__init__(df=df, weight=weight)
        self.name = 'ema'
        f.add_ema(df=df, p=fast)
        f.add_ema(df=df, p=slow)
        colfast, colslow = f'ema{fast}', f'ema{slow}'

        df['emaspread'] = round((df[colfast] - df[colslow]) / ((df[colfast] + df[colslow]) / 2) , 6)
        df['ema_trend'] = np.where(df[colfast] > df[colslow], 1, -1)

        c = self.get_c(maxspread=0.1)
        df['ema_conf'] = self.ema_exp(x=df.emaspread, c=c)

        self.trendseries = df.ema_trend

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
    def __init__(self, df, weight=1, p=50, slope=5):
        super().__init__(df=df, weight=weight)
        self.name = 'ema_Slope'
        f.add_ema(df=df, p=p)
        df['ema_slope'] = np.where(np.roll(df['ema{}'.format(p)], slope, axis=0) < df['ema{}'.format(p)], 1, -1)
        df.loc[:p + slope, 'ema_slope'] = np.nan

        self.trendseries = df.ema_slope

    def final(self, side, c):
        if side * c.ema_slope == 1:
            conf = 1.5
        else:
            conf = 0.5

        return conf * self.weight     

class Volatility(Signal):
    def __init__(self, df, weight=1, norm=(0.004,0.024)):
        super().__init__(df=df, weight=weight)
        self.name = 'volatility'

        df['maxhigh'] = df.High.rolling(48).max()
        df['minlow'] = df.Low.rolling(48).min()
        df['spread'] = abs(df.maxhigh - df.minlow) / df[['maxhigh', 'minlow']].mean(axis=1)

        df['emavty'] = df.spread.ewm(span=60, min_periods=60).mean()
        df['smavty'] = df.spread.rolling(300).mean()
        df['norm_ema'] = np.interp(df.emavty, (0, 0.25), (norm[0], norm[1]))
        df['norm_sma'] = np.interp(df.smavty, (0, 0.25), (norm[0], norm[1]))
        # df['normtp'] = np.interp(df.smavty, (0, 0.4), (0.3, 3)) # only Strat_Chop

        df.drop(columns=['maxhigh', 'minlow'], inplace=True)

    def final(self, c):
        # return self.df.norm_ema[i]
        return c.norm_ema

class Trend(Signal):
    def __init__(self, df, signals, speed, offset=1):
        super().__init__(df=df)
        # accept 1 to n series of trend signals, eg 1 or -1
        # sum signals > positive = 1, negative = -1, neutral = 0
        # df['temp'] = np.sum(signals, axis=0)
        # df['trend'] = np.where(df.temp == 0, 0, np.where(df.temp > 0, 1, -1))
        # ^didn't work, just use ema_trend for now
        df['temp'] = np.nan
        df['trend'] = df.ema_trend

        # set trade high/low in period prices
        against, wth, neutral = speed[0], speed[1], int(np.average(speed))

        # max highs
        df['mhw'] = df.High.rolling(wth).max().shift(offset)
        df['mha'] = df.High.rolling(against).max().shift(offset)
        df['mhn'] = df.High.rolling(neutral).max().shift(offset)

        # min lows
        df['mla'] = df.Low.rolling(wth).min().shift(offset)
        df['mlw'] = df.Low.rolling(against).min().shift(offset)
        df['mln'] = df.Low.rolling(neutral).min().shift(offset)

        df['pxhigh'] = np.where(df.trend == 0, df.mhn, np.where(df.trend == 1, df.mha, df.mhw))
        df['pxlow'] = np.where(df.trend == 0, df.mln, np.where(df.trend == -1, df.mlw, df.mla))
        
        df.drop(columns=['mha', 'mhw', 'mla', 'mlw', 'mhn', 'mln', 'temp'], inplace=True)

    def final(self, c):
        return c.trend
