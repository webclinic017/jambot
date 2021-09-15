import io
import logging
import re
import traceback

import google.cloud.logging as gcl
from google.auth.credentials import Credentials
from google.oauth2 import service_account

from jambot.config import AZURE_WEB

try:
    import colored_traceback
    import colorlog
    from colored_traceback import Colorizer

    # color tracebacks in terminal - uncaught exceptions in scripts only, not logging
    colored_traceback.add_hook(style='jayme', always=True)
    Formatter = colorlog.ColoredFormatter
except ModuleNotFoundError:
    # running on azure
    Formatter = logging.Formatter


# simple colors
_palette = dict(
    black=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    cyan=36,
    white=37,
    underline=4,
    reset=0)

# create full color codes as a dict comp
palette = {k: f'\033[{color_code}m' for k, color_code in _palette.items()}


class ColoredFormatter(Formatter):
    """Custom logging Formatter to print colored tracebacks and log level messages
    """

    def __init__(self, fmt: str, *args, **kw) -> None:
        log_colors = dict(
            DEBUG='cyan',
            INFO='green',
            WARNING='yellow',
            ERROR='red',
            CRITICAL='red,bg_white')

        # always add log_color before format
        fmt = f'%(log_color)s{fmt}'

        super().__init__(fmt, log_colors=log_colors, *args, **kw)
        self.colorizer = Colorizer(style='jayme')

    def colorize_traceback(self, type, value, tb) -> str:
        """
        Copied from colored_traceback Colorizer
        - just return and print to io instead of write to stderr so logging message prints first
        """

        import pygments.lexers
        tb_text = ''.join(traceback.format_exception(type, value, tb))
        lexer = pygments.lexers.get_lexer_by_name('pytb', stripall=True)
        tb_colored = pygments.highlight(tb_text, lexer, self.colorizer.formatter)
        # self.stream.write(tb_colored)
        return tb_colored

    def formatException(self, ei) -> str:
        sio = io.StringIO()

        tb_colored = self.colorize_traceback(*ei)
        print(tb_colored, file=sio)

        s = sio.getvalue()
        sio.close()

        if s[-1:] == '\n':
            s = s[:-1]

        return s

    def formatMessage(self, record: logging.LogRecord) -> str:
        message = super().formatMessage(record)
        return highlight_filepath(message)


class Logger(logging.Logger):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.propagate = False  # this prevents duplicate outputs
        self.setLevel(logging.DEBUG)

        if not AZURE_WEB:
            # local app, use colored formatter
            Formatter = ColoredFormatter
        else:
            Formatter = logging.Formatter

        # Console/stream handler
        fmt_stream = '%(levelname)-7s %(lineno)-4d %(name)-20s %(message)s'
        formatter = Formatter(fmt_stream)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        if not self.handlers:
            self.addHandler(sh)

            if AZURE_WEB:
                add_google_logging_handler(self)


def get_creds(scopes: list = None) -> Credentials:
    """Create google oauth2 credentials
    - NOTE duplicated from jambot.utils.google to avoid imports

    Parameters
    ----------
    scopes : list, optional
        default None

    Returns
    -------
    Credentials
    """
    from jambot.utils.secrets import SecretsManager
    m = SecretsManager('google_creds.json').load
    return service_account.Credentials.from_service_account_info(m, scopes=scopes)


def get_google_logging_client() -> gcl.Client:
    """Create google logging client

    Returns
    -------
    google.logging.cloud.Client
        logging client to add handler to Python logger
    """
    return gcl.Client(credentials=get_creds())


def add_google_logging_handler(log: logging.Logger) -> None:
    """Add gcl handler to Python logger

    Parameters
    ----------
    log : logging.Logger
    """
    gcl_client = get_google_logging_client()
    handler = gcl_client.get_default_handler()
    handler.setLevel(logging.INFO)
    log.addHandler(handler)


def getlog(name):
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
    >>> from smseventlog import getlog
    >>> log = getlog(__name__)
    """
    name = '.'.join(str(name).split('.')[1:])

    # cant set name to nothing or that calls the ROOT logger
    if name == '':
        name = 'base'

    return Logger(name)


def highlight_filepath(s: str, color: str = 'blue') -> str:
    """Highlight filepath string for output in terminal

    - TODO add https for url

    Parameters
    ----------
    s : str
        string to search for filepaths
    color : str, optional
        default 'blue'

    Returns
    -------
    str
        input string with filepaths colored
    """
    # try to match previous color
    # \x1b[32m
    expr = r'\x1b\[\d+m'
    match = re.search(expr, str(s))
    reset = match[0] if match else palette['reset']

    # stop at first backslash \ (color code)
    expr = r'(http.*|\/.*\/[^\s\\]*)'
    return re.sub(expr, f'{palette[color]}\\1{reset}', str(s))
