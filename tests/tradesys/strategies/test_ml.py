from jambot.tradesys.strategies.ml import Strategy

from .__init__ import *


def test_init(symbol: Symbol):
    strat = Strategy(symbol=symbol)


def test_strat(symbol: Symbol):
    strat = Strategy(symbol=symbol)
