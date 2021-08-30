"""
Loading dataframe
"""

from pathlib import Path

import numpy as np
import pandas as pd


def default_df():
    """Get simple default df for testing"""
    p = Path('df.csv')

    # NOTE not dry - defined in database.get_db()
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    dtypes = {c: np.float32 for c in cols[1:]}
    dtypes['volume'] = pd.Int64Dtype()

    return pd \
        .read_csv(
            p,
            parse_dates=['timestamp'],
            index_col='timestamp') \
        .astype(dtypes)
