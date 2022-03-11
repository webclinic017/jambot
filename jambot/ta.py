from importlib.util import find_spec

import pandas as pd
from pandas_ta.overlap import ma
from pandas_ta.utils import zero
from pandas_ta.volatility import atr

TALIB_EXISTS = not find_spec('talib') is None


def adx(
        df: pd.DataFrame,
        length: int = 14,
        lensig=None,
        scalar: int = 100,
        mamode=None,
        drift: int = 1,
        offset: int = 0,
        **kwargs) -> pd.Series:
    """Indicator: ADX"""
    # Validate Arguments

    lensig = lensig if lensig and lensig > 0 else length
    mamode = mamode if isinstance(mamode, str) else 'rma'
    high, low, close = df.high, df.low, df.close

    if TALIB_EXISTS:
        from talib import ADX
        adx = ADX(high, low, close, timeperiod=length)
    else:
        atr_ = atr(high=high, low=low, close=close, length=length)

        up = high - high.shift(drift)  # high.diff(drift)
        dn = low.shift(drift) - low    # low.diff(-drift).shift(drift)

        pos = ((up > dn) & (up > 0)) * up
        neg = ((dn > up) & (dn > 0)) * dn

        pos = pos.apply(zero)
        neg = neg.apply(zero)

        k = scalar / atr_
        dmp = k * ma(mamode, pos, length=length)
        dmn = k * ma(mamode, neg, length=length)

        dx = scalar * (dmp - dmn).abs() / (dmp + dmn)
        adx = ma(mamode, dx, length=lensig)

    # Offset
    if offset != 0:
        # dmp = dmp.shift(offset)
        # dmn = dmn.shift(offset)
        adx = adx.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        adx.fillna(kwargs['fillna'], inplace=True)
        # dmp.fillna(kwargs["fillna"], inplace=True)
        # dmn.fillna(kwargs["fillna"], inplace=True)
    if 'fill_method' in kwargs:
        adx.fillna(method=kwargs['fill_method'], inplace=True)
        # dmp.fillna(method=kwargs["fill_method"], inplace=True)
        # dmn.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    adx.name = f'adx_{lensig}'
    # dmp.name = f"DMP_{length}"
    # dmn.name = f"DMN_{length}"

    return adx
