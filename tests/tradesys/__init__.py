import pandas as pd

from jambot import data
from jambot.tradesys.base import Clock
from jambot.tradesys.symbols import Symbol

from ..__init__ import *


@fixture(scope='session')
def default_df(symbol: Symbol, exch_name: str) -> pd.DataFrame:
    """Get default OHLC data for single symbol"""

    # NOTE temp kinda ugly
    if symbol == 'BTCUSD':
        symbol = 'XBTUSD'

    return data.default_df(symbol=symbol, exch_name=exch_name)


@fixture
def clock(default_df: pd.DataFrame) -> Clock:
    """Get Clock timer obj"""
    return Clock(df=default_df)
