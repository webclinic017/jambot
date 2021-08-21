import json  # noqa
import logging  # noqa
import os  # noqa
import re  # noqa
import sys  # noqa
import time  # noqa
from collections import defaultdict as dd  # noqa
from datetime import date  # noqa
from datetime import datetime as dt  # noqa
from datetime import timedelta as delta  # noqa
from datetime import timezone as tz  # noqa
from pathlib import Path  # noqa
from timeit import default_timer as timer  # noqa
from typing import *  # noqa

import numpy as np  # noqa
import pandas as pd  # noqa
import pypika as pk  # noqa

try:
    from icecream import ic  # noqa
    from IPython.display import display  # noqa
    ic.configureOutput(prefix='')
except ModuleNotFoundError:
    pass


# Set environments
AZURE_LOCAL = not os.getenv('AZURE_FUNCTIONS_ENVIRONMENT') is None
AZURE_WEB = not os.getenv('WEBSITE_SITE_NAME') is None
SYMBOL = 'XBTUSD'

if AZURE_LOCAL or AZURE_WEB:
    sys.modules['__main__'] = None  # this isn't set properly when run in azure and throws KeyError


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

        import jambot.utils.google as gg
        gg.add_google_logging_handler(log)

    return log
