from datetime import datetime as dt
from pathlib import Path

from jambot import AZURE_WEB

p_proj = Path(__file__).parent  # jambot python files
p_root = p_proj.parent  # root folter
p_res = p_proj / '_res'
p_sec = p_res / 'secrets'

# set data dir for local vs azure
p_data = p_root / 'data' if not AZURE_WEB else Path.home() / 'data'

config = dict(
    d_lower=dt(2017, 1, 1),
    interval=15,
    symbol='XBTUSD',
    drop_cols=['open', 'high', 'low', 'close', 'volume', 'ema_10', 'ema_50', 'ema_200'],
    signalmanager_kw=dict(slope=1, sum=12)
)

colors = dict(
    lightblue='#6df7ff',
    lightred='#ff6d6d',
    darkgrey='#a0a0a0',
    lightyellow='#FFFFCC'
)
