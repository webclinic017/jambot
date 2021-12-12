from datetime import date  # noqa
from datetime import datetime as dt  # noqa
from datetime import timedelta as delta  # noqa
from datetime import timezone as tz  # noqa

from jgutils.logger import getlog  # noqa

from jambot.config import SYMBOL  # noqa

try:
    from icecream import ic  # noqa
    from IPython.display import display  # noqa
    ic.configureOutput(prefix='')
except ModuleNotFoundError:
    display = lambda *args: args
