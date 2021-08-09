from jambot import data
from jambot.tradesys.base import Clock

from ..__init__ import *


@fixture
def clock():
    df = data.default_df()
    clock = Clock(df=df)
    return clock
