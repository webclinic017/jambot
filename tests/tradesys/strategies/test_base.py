from jambot.tradesys.strategies.base import StrategyBase

from .__init__ import *


def init_strategy():
    strat = StrategyBase(symbol=SYMBOL)
