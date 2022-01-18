import os
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import *

import yaml

# Set environments
AZURE_LOCAL = not os.getenv('AZURE_FUNCTIONS_ENVIRONMENT') is None
AZURE_WEB = not os.getenv('WEBSITE_SITE_NAME') is None
SYMBOL = 'XBTUSD'

if AZURE_LOCAL or AZURE_WEB:
    sys.modules['__main__'] = None  # type: ignore - this isn't set properly when run in azure and throws KeyError


p_proj = Path(__file__).parent  # jambot (python module)
p_root = p_proj.parent  # root folter
p_res = p_proj / '_res'
p_sec = p_res / 'secrets'
os.environ['p_secret'] = str(p_sec)
os.environ['p_unencrypt'] = str(p_proj / '_unencrypted')
p_cfg = p_res / 'model_config.yaml'  # dynamic model config file

# set data dir for local vs azure
p_data = p_root / 'data' if not AZURE_WEB else Path.home() / 'data'
p_ftr = p_data / 'feather'

INTERVAL = 15
SYMBOL = 'XBTUSD'
D_LOWER = dt(2017, 1, 1)
D_SPLIT = dt(2021, 2, 1)
# D_SPLIT = dt(2020, 1, 1)
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


def dynamic_cfg(symbol: str = SYMBOL, keys: Union[List[str], Dict[str, str]] = None) -> Dict[str, Any]:
    """Get dynamic config values per symbol
    - TODO find a clean way to manage static/dynamic keys

    Parameters
    ----------
    symbol : str
        eg XBTUSD
    keys: Union[List[str], Dict[str, str]]
        pass in keys to filter by, or dict of keys with conv value, default None

    Returns
    -------
    Dict[str, Any]
    """
    # TODO temp solution for BTCUSD
    symbol = 'XBTUSD' if symbol == 'BTCUSD' else symbol

    with open(p_cfg, 'r') as file:
        m = yaml.full_load(file)[symbol]

    # filter to only requested keys
    if not keys is None:
        if isinstance(keys, list):
            m = {k: v for k, v in m.items() if k in keys}
        elif isinstance(keys, dict):
            m = {keys[k]: v for k, v in m.items() if k in keys.keys()}

    return m
