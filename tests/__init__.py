import pytest  # noqa
from pytest import fixture, mark, raises  # noqa

from jambot import SYMBOL, Num  # noqa
from jambot import functions as f  # noqa
from jambot.tradesys.symbols import Symbol


@mark.skip()
def mul(val: Num, symbol: Symbol) -> Num:
    """Multiply prices/qtys for different symbols"""
    if symbol in ('XBTUSD', 'BTCUSD'):
        mult = 100
    elif symbol == 'BNBUSDT':
        mult = 5
    else:
        mult = 5

    return val * mult
