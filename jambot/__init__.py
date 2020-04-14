import sys
import logging
import os

azure_env = os.getenv("AZURE_FUNCTIONS_ENVIRONMENT")

if not (azure_env or 'linux' in sys.platform):
    # when not running from packaged app, import all libraries for easy access in interactive terminal
    import json
    from datetime import (datetime as dt, timedelta as delta)
    from pathlib import Path
    from time import time
    from timeit import Timer
    import cProfile

    import pandas as pd
    import numpy as np
    import pypika as pk
    import plotly.offline as py
    from joblib import Parallel, delayed
    import yaml
    import sqlalchemy as sa

    from . import (
        functions as f,
        livetrading as live,
        backtest as bt)
    from .database import db