"""
Loading dataframe
"""

import pandas as pd

from jambot import SYMBOL
from jambot import config as cf
from jambot import getlog

log = getlog(__name__)


def default_df(symbol: str = SYMBOL) -> pd.DataFrame:
    """Get simple default df for testing"""
    log.info(f'Loading df for "{symbol}" from file')
    p = cf.p_data / f'feather/df_{symbol.lower()}.ftr'
    return pd.read_feather(p) \
        .set_index('timestamp')
