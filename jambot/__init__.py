import json
import logging
import os
import re
import sys
import time
from collections import defaultdict as dd
from datetime import date
from datetime import datetime as dt
from datetime import timedelta as delta
from datetime import timezone as tz
from pathlib import Path
from timeit import default_timer as timer
from typing import *

import numpy as np
import pandas as pd
import pypika as pk

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


def getlog(name: str) -> logging.Logger:
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
    >>> from jambot import getlog
        log = getlog(__name__)
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
