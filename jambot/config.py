import os
import sys
from datetime import datetime as dt
from pathlib import Path

# Set environments
AZURE_LOCAL = not os.getenv('AZURE_FUNCTIONS_ENVIRONMENT') is None
AZURE_WEB = not os.getenv('WEBSITE_SITE_NAME') is None
SYMBOL = 'XBTUSD'

if AZURE_LOCAL or AZURE_WEB:
    sys.modules['__main__'] = None  # this isn't set properly when run in azure and throws KeyError


p_proj = Path(__file__).parent  # jambot (python module)
p_root = p_proj.parent  # root folter
p_res = p_proj / '_res'
p_sec = p_res / 'secrets'
os.environ['p_secret'] = str(p_sec)
os.environ['p_unencrypt'] = str(p_proj / '_unencrypted')

# set data dir for local vs azure
p_data = p_root / 'data' if not AZURE_WEB else Path.home() / 'data'
p_ftr = p_data / 'feather'

INTERVAL = 15
SYMBOL = 'XBTUSD'
D_LOWER = dt(2017, 1, 1)
# D_SPLIT = dt(2021, 2, 1)
D_SPLIT = dt(2021, 8, 1)
DROP_COLS = ['open', 'high', 'low', 'close', 'volume', 'ema10', 'ema50', 'ema200']
FILTER_FIT_QUANTILE = 0.55

SIGNALMANAGER_KW = dict(
    slope=[1, 4, 8, 16, 32, 64],
    sum=[12, 24, 96])

WEIGHTSMANAGER_KW = dict(
    n_periods=8,
    weight_linear=True)

COLORS = dict(
    lightblue='#6df7ff',
    lightred='#ff6d6d',
    darkgrey='#a0a0a0',
    lightyellow='#FFFFCC')
