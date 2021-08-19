"""General Functions module - don't rely on any other modules from jambot"""

import json
import math
import pickle
import re
import time
from datetime import date
from datetime import datetime as dt
from datetime import timedelta as delta
from pathlib import Path
from sys import platform
from typing import Any, Union

import pandas as pd
from dateutil.parser import parse


def check_path(p: Path) -> None:
    """Create bath if doesn't exist"""
    if not p.exists():
        p.mkdir(parents=True)


def left_merge(df: pd.DataFrame, df_right: pd.DataFrame) -> pd.DataFrame:
    """Convenience func to left merge df on index

    Parameters
    ----------
    df : pd.DataFrame
    df_right : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        df with df_right merged
    """
    return df \
        .merge(
            right=df_right,
            how='left',
            left_index=True,
            right_index=True)


def as_list(items: Any) -> list:
    """Check item(s) is list, make list if not"""
    if not isinstance(items, list):
        items = [items]

    return items


def set_self(m: dict, exclude: Union[tuple, str] = ()):
    """Convenience func to assign an object's func's local vars to self"""
    if not isinstance(exclude, tuple):
        exclude = (exclude, )
    exclude += ('__class__', 'self')  # always exclude class/self
    obj = m.get('self', None)  # self must always be in vars dict

    if obj is None:
        return

    for k, v in m.items():
        if not k in exclude:
            setattr(obj, k, v)


def inverse(m: dict) -> dict:
    """Return inverse of dict"""
    return {v: k for k, v in m.items()}


def pretty_dict(m: dict, html=False, prnt=True) -> str:
    """Print pretty dict converted to newlines
    Paramaters
    ----
    m : dict
    html: bool
        Use <br> instead of html
    Returns
    -------
    str
        'Key 1: value 1
        'Key 2: value 2"
    """
    s = json.dumps(m, indent=4)
    newline_char = '\n' if not html else '<br>'

    # remove these chars from string
    remove = '}{\'"[]'
    for char in remove:
        s = s.replace(char, '')

        # .replace(', ', newline_char) \
    s = s \
        .replace(',\n', newline_char)

    # remove leading and trailing newlines
    s = re.sub(r'^[\n]', '', s)
    s = re.sub(r'\s*[\n]$', '', s)

    if prnt:
        print(s)
    else:
        return s


def df_dict(m: dict, colname=None, prnt=True):
    """Quick display of dataframe from dict

    Parameters
    ----------
    m : dict
        dictionary to display
    colname : str, optional
    prnt : bool, optional
    """
    from IPython.display import display

    colname = colname or 'col1'
    df = pd.DataFrame.from_dict(m, orient='index', columns=[colname])

    if prnt:
        display(df)
    else:
        return df


def filter_df(dfall, symbol):
    return dfall[dfall.symbol == symbol].reset_index(drop=True)


def filter_cols(df: pd.DataFrame, expr: str = '.') -> list:
    """Return list of cols in df based on regex expr

    Parameters
    ----------
    df : pd.DataFrame
    expr : str, optional
        default '.'

    Returns
    -------
    list
        list of cols which match expression
    """
    return [c for c in df.columns if re.search(expr, c)]


def select_cols(df: pd.DataFrame, expr: str = '.', include: list = None) -> pd.DataFrame:
    """Filter df cols based on regex

    Parameters
    ----------
    df : pd.DataFrame
    expr : str, optional
        regex expr, by default '.'

    Returns
    -------
    pd.DataFrame
    """
    cols = filter_cols(df, expr)

    # include other cols
    if not include is None:
        include = as_list(include)
        cols += include

    return df[cols]


def drop_cols(df: pd.DataFrame, expr: str = '.') -> pd.DataFrame:
    """Filter df cols based on regex

    Parameters
    ----------
    df : pd.DataFrame
    expr : str, optional
        regex expr, by default '.'

    Returns
    -------
    pd.DataFrame
    """
    return df.drop(columns=filter_cols(df, expr))


def clean_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return cols if they exist in dataframe"""
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def safe_drop(df: pd.DataFrame, cols: Union[str, list], do: bool = True) -> pd.DataFrame:
    """Drop columns from dataframe if they exist

    Parameters
    ----------
    df : pd.DataFrame
    cols : Union[str, list]
        list of cols or str

    Returns
    -------
    pd.DataFrame
        df with cols dropped
    """
    if not do:
        return df

    cols = [c for c in as_list(cols) if c in df.columns]
    return df.drop(columns=cols)


def reduce_dtypes(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """Change dtypes from {select: to}

    Parameters
    ----------
    df : pd.DataFrame
    dtypes : dict
        dict eg {np.float32: np.float16}

    Returns
    -------
    pd.DataFrame
        df with dtypes changed
    """
    dtype_cols = {}
    for d_from, d_to in dtypes.items():
        dtype_cols |= {c: d_to for c in df.select_dtypes(d_from).columns}

    return df.astype(dtype_cols)

# DATETIME


def check_date(d):
    if type(d) is str:
        return parse(d)
    else:
        return d


def startvalue(startdate):
    return startdate + delta(days=-25)


def enddate(startdate, rng):
    return startdate + delta(days=rng)


def time_format(hrs=False, mins=False, secs=False):
    if secs:
        return '%Y-%m-%d %H:%M:%S'
    elif mins:
        return '%Y-%m-%d %H:%M'
    elif hrs:
        return '%Y-%m-%d %H'
    else:
        return '%Y-%m-%d'


def print_time(start):
    end = time.time()
    duration = end - start
    if duration < 60:
        ans = '{:.3f}s'.format(duration)
    else:
        mins = duration / 60
        secs = duration % 60
        ans = '{:.0f}m {:.3f}s'.format(mins, secs)

    print(ans)


def read_csv(startdate, daterange, symbol=None):
    p = Path.cwd().parent / 'Testing/df.csv'
    df = pd.read_csv(p, parse_dates=['timestamp'], index_col=0)

    if not symbol is None:
        df = filter_df(dfall=df, symbol=symbol)

    mask = ((df['timestamp'] >= startvalue(startdate)) & (df['timestamp'] <= enddate(startdate, daterange)))
    df = df.loc[mask].reset_index(drop=True)
    return df


def percent(val):
    return '{:.2%}'.format(val)


def round_down(n: Union[int, float], nearest: int = 100) -> int:
    """Round down to nearest value (used for round contract sizes to nearest 100)

    Parameters
    ----------
    n : Union[int, float]
        number to round
    nearest : int, optional
        value to round to, by default 100

    Returns
    -------
    int
        number rounded to nearest
    """
    return int(math.floor(n / nearest)) * nearest


def price_format(altstatus=False):
    ans = '{:,.0f}' if not altstatus else '{:,.0f}'
    return ans


def get_price(pnl: float, entry_price: float, side: int) -> float:
    """Get price at percentage offset

    Parameters
    ----------
    pnl : float
        percent offset from entry_price
    entry_price : float
        price to offset from
    side : int
        order side

    Returns
    -------
    float
        price
    """
    if side == 1:
        price = pnl * entry_price + entry_price
    elif side == -1:
        price = entry_price / (1 + pnl)

    # NOTE this will need to change for different symbols
    return round_down(n=price, nearest=0.5)


def get_pnl_xbt(qty: int, entry_price: float, exit_price: float, isaltcoin: bool = False) -> float:
    """Get profit/loss in base currency (xbt) from entry/exit

    Parameters
    ----------
    qty : int, optional
        quantity of contracts
    entry_price : float, optional
    exit_price : float, optional
    isaltcoin : bool, optional
        alts calculated opposite, default False
        - NOTE this should all be restructured to be relative to a base currency somehow

    Returns
    -------
    float
        amount of xbt gained/lossed
    """
    if 0 in (entry_price, exit_price):
        return 0
    elif not isaltcoin:
        return round(qty * (1 / entry_price - 1 / exit_price), 8)
    elif isaltcoin:
        return round(qty * (exit_price - entry_price), 8)


def get_pnl(side, entry_price, exit_price):
    if 0 in (entry_price, exit_price):
        return 0
    elif side == 1:
        return round((exit_price - entry_price) / entry_price, 4)
    else:
        return round((entry_price - exit_price) / exit_price, 4)


def get_contracts(xbt, leverage, price, side, is_altcoin=False):
    if not is_altcoin:
        return int(xbt * leverage * price * side)
    else:
        return int(xbt * leverage * (1 / price) * side)


def side(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def useful_keys(orders):
    """return only useful keys from bitmex orders in dict form
    - NOTE don't think this is needed anymore
    """
    keys = ('symbol', 'clOrdID', 'side', 'price', 'stopPx', 'ordType',
            'execInst', 'ordStatus', 'qty', 'name', 'manual', 'orderID')

    if not isinstance(orders, list):
        islist = False
        orders = [orders]
    else:
        islist = True

    result = [{key: o[key] for key in o.keys() if key in keys} for o in orders]

    if islist:
        return result
    else:
        return result[0]


def col(df, col):
    return df.columns.get_loc(col)


def discord(msg: str, channel: str = 'jambot') -> None:
    """Send message to discord channel

    Parameters
    ----------
    msg : str
    channel : str, optional
        discord channel, default 'jambot'
    """
    from discord import RequestsWebhookAdapter, Webhook

    from jambot.utils.secrets import SecretsManager

    r = SecretsManager('discord.csv').load.set_index('channel').loc[channel]
    if channel == 'err':
        msg += '@here'

    # Create webhook
    webhook = Webhook.partial(r.id, r.token, adapter=RequestsWebhookAdapter())

    # split into strings of max 2000 char for discord
    n = 2000
    out = [(msg[i:i + n]) for i in range(0, len(msg), n)]

    for msg in out:
        webhook.send(msg)


def send_error(msg: str = '', prnt: bool = False) -> None:
    import traceback
    err = traceback.format_exc().replace('Traceback (most recent call last):\n', '')

    if not msg == '':
        err = '{}:\n{}'.format(msg, err).replace(':\nNoneType: None', '')

    err = '*------------------*\n{}'.format(err)

    if prnt or not 'linux' in platform:
        print(err)
    else:
        discord(msg=err, channel='err')


def get_google_sheet():
    import pygsheets
    from google.oauth2 import service_account

    from jambot.utils.secrets import SecretsManager
    m = SecretsManager('gsheets.json').load
    SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    my_credentials = service_account.Credentials.from_service_account_info(m, scopes=SCOPES)

    # easiest way loading from json file
    # return pygsheets.authorize(service_account_file=p).open('Jambot Settings')
    return pygsheets.authorize(custom_credentials=my_credentials).open('Jambot Settings')


def get_offset(interval: int = 1) -> delta:
    """Get single period offset for 1hr or 15m

    Parameters
    ----------
    interval : int, optional
        default 1 hr

    Returns
    -------
    datetime.timedelta
        offset for selected interval
    """
    return {1: delta(hours=1), 15: delta(minutes=15)}.get(interval)


def date_to_dt(d: date) -> dt:
    return dt.combine(d, dt.min.time())


def timenow(interval: int = 1) -> dt:
    """Get current utc time rounded down to nearest 1hr/15min interval

    Parameters
    ----------
    interval : int, optional
        default 1

    Returns
    -------
    dt
        datetime rounded to interval
    """
    if interval == 1:
        return dt.utcnow().replace(microsecond=0, second=0, minute=0)
    elif interval == 15:
        return round_minutes(dt=dt.utcnow(), resolution=15).replace(microsecond=0, second=0)


def round_minutes(dt, resolution):
    new_minute = (dt.minute // resolution) * resolution
    return dt + delta(minutes=new_minute - dt.minute)


def save_pickle(obj: object, p: Path, name: str):
    """Save object to pickle file"""
    p = p / f'{name}.pkl'
    with open(p, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(p: Path) -> object:
    """Load pickle from file"""
    with open(p, 'rb') as file:
        return pickle.load(file)


def remove_bad_chars(w: str):
    """Remove any bad chars " : < > | . \ / * ? in string to make safe for filepaths"""  # noqa
    return re.sub(r'[":<>|.\\\/\*\?]', '', str(w))


def to_snake(s: str):
    """Convert messy camel case to lower snake case

    Parameters
    ----------
    s : str
        string to convert to special snake case

    Examples
    --------
    """
    s = remove_bad_chars(s).strip()  # get rid of /<() etc
    s = re.sub(r'[\]\[()]', '', s)  # remove brackets/parens
    s = re.sub(r'[\n-]', '_', s)  # replace newline/dash with underscore
    s = re.sub(r'[%]', 'pct', s)
    s = re.sub(r"'", '', s)

    # split on capital letters
    expr = r'(?<!^)((?<![A-Z])|(?<=[A-Z])(?=[A-Z][a-z]))(?=[A-Z])'

    return re \
        .sub(expr, '_', s) \
        .lower() \
        .replace(' ', '_') \
        .replace('__', '_')


def lower_cols(df):
    """Convert df columns to snake case and remove bad characters"""
    is_list = False

    if isinstance(df, pd.DataFrame):
        cols = df.columns
    else:
        cols = df
        is_list = True

    m_cols = {col: to_snake(col) for col in cols}

    if is_list:
        return list(m_cols.values())

    return df.pipe(lambda df: df.rename(columns=m_cols))


def parse_datecols(df, format=None):
    """Convert any columns with 'date' or 'time' in header name to datetime"""
    datecols = list(filter(lambda x: any(s in x.lower()
                    for s in ('date', 'time')), df.columns))
    df[datecols] = df[datecols].apply(
        pd.to_datetime, errors='coerce', format=format)
    return df
