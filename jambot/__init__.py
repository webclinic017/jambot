import logging
import os
import sys
from typing import *

try:
    from icecream import ic
    from IPython.display import display
    ic.configureOutput(prefix='')
except ModuleNotFoundError:
    pass


# Set environments
AZURE_LOCAL = not os.getenv('AZURE_FUNCTIONS_ENVIRONMENT') is None
AZURE_WEB = not os.getenv('WEBSITE_SITE_NAME') is None
SYMBOL = 'XBTUSD'

if not (AZURE_WEB or 'linux' in sys.platform):
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
