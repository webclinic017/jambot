"""
Loading dataframe
"""

from pathlib import Path

import pandas as pd


def default_df():
    """Get simple default df for testing"""
    p = Path('df.csv')

    return pd \
        .read_csv(
            p,
            parse_dates=['Timestamp'],
            index_col='Timestamp')
