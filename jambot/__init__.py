import logging
import os
import sys
from typing import *

from icecream import ic

try:
    from IPython.display import display
except ModuleNotFoundError:
    pass

ic.configureOutput(prefix='')

AZURE_ENV = os.getenv('AZURE_FUNCTIONS_ENVIRONMENT')
SYMBOL = 'XBTUSD'

if not (AZURE_ENV or 'linux' in sys.platform):
    # when not running from packaged app, import all libraries for easy access in interactive terminal
    import cProfile
    import json
    import time
    from collections import defaultdict as dd
    from datetime import datetime as dt
    from datetime import timedelta as delta
    from datetime import timezone as tz
    from pathlib import Path
    from timeit import Timer

    import numpy as np
    import pandas as pd
    import plotly.offline as py
    import pypika as pk
    import sqlalchemy as sa
    import yaml
    from joblib import Parallel, delayed

    # from . import (
    #     functions as f,
    #     livetrading as live,
    #     backtest as bt)
    # from .database import db


def getlog(name: str):
    """Create logger object with predefined stream handler & formatting

    Parameters
    ----------
    name : str
        module __name__

    Returns
    -------
    logging.logger

    Examples
    --------
    >>> from .__init__ import getlog
    >>> log = getlog(__name__)
    """
    fmt_stream = logging.Formatter('%(levelname)-7s %(lineno)-4d %(name)-26s %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt_stream)

    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        log.addHandler(sh)

    return log
