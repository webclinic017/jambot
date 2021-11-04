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
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'funding_rate']
    dtypes = {c: np.float32 for c in cols[1:]}
    dtypes['volume'] = pd.Int64Dtype()
    dtypes['funding_rate'] = np.float32

    df = pd \
        .read_csv(
            p,
            parse_dates=['timestamp'],
            index_col='timestamp')

    dtypes = {k: v for k, v in dtypes.items() if k in df.columns}
    return df.astype(dtypes)
