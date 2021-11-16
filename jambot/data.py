"""
Loading dataframe
"""

import pandas as pd

from jambot import config as cf


def default_df():
    """Get simple default df for testing"""
    p = cf.p_data / 'feather/df.ftr'
    return pd.read_feather(p) \
        .set_index('timestamp')
