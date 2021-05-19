from jambot.tradesys.strategies.ml import Strategy

from .__init__ import *


def test_init():
    strat = Strategy(symbol=SYMBOL)
