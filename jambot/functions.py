"""General Functions module - don't rely on any other modules from jambot"""

import json
import os
import pickle
import re
import time
from datetime import date
from datetime import datetime as dt
from datetime import timedelta as delta
from pathlib import Path
from sys import platform
from typing import Any

import pandas as pd
import pyodbc
import pypika as pk
import yaml
from dateutil.parser import parse
from pypika import functions as fn

from jambot import AZURE_WEB
from jambot.utils.secrets import SecretsManager

p_proj = Path(__file__).parent  # jambot python files
p_root = p_proj.parent  # root folter
p_res = p_proj / '_res'
p_sec = p_res / 'secrets'

# set data dir for local vs azure
p_data = p_root / 'data' if not AZURE_WEB else Path.home() / 'data'


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


def set_self(m, prnt=False, exclude=()):
    """Convenience func to assign an object's func's local vars to self"""
    # if not isinstance(exclude, tuple): exclude = (exclude, )
    exclude += ('__class__', 'self')  # always exclude class/self
    obj = m.get('self', None)  # self must always be in vars dict

    if obj is None:
        return

    for k, v in m.items():
        # if prnt:
        #     print(f'\n\t{k}: {v}')

        if not k in exclude:
            setattr(obj, k, v)


def filter_df(dfall, symbol):
    return dfall[dfall.Symbol == symbol].reset_index(drop=True)


def filter_cols(df: pd.DataFrame, expr: str = '.', include: list = None) -> pd.DataFrame:
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
    cols = [c for c in df.columns if re.search(expr, c)]

    # include other cols
    if not include is None:
        include = as_list(include)
        cols += include

    return df[cols]

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
    df = pd.read_csv(p, parse_dates=['Timestamp'], index_col=0)

    if not symbol is None:
        df = filter_df(dfall=df, symbol=symbol)

    mask = ((df['Timestamp'] >= startvalue(startdate)) & (df['Timestamp'] <= enddate(startdate, daterange)))
    df = df.loc[mask].reset_index(drop=True)
    return df


def percent(val):
    return '{:.2%}'.format(val)


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
        return pnl * entry_price + entry_price
    elif side == -1:
        return entry_price / (1 + pnl)


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
        # NOTE this should all be restructured to be relative to a base currency somehow

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
    # return only useful keys from bitmex orders in dict form
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


def key(symbol, name, side, ordtype):
    # if ordtype == 'Stop':
    #     side *= -1

    sidestr = 'long' if side == 1 else 'short'

    return '{}-{}-{}'.format(symbol, name.lower(), sidestr)


def col(df, col):
    return df.columns.get_loc(col)


def binance_creds():
    return
    p = p_sec / 'binance.yaml'
    with open(p) as file:
        return yaml.full_load(file)


def discord(msg, channel='jambot'):
    import discord
    import requests
    from discord import File, RequestsWebhookAdapter, Webhook

    p = p_sec / 'discord.csv'
    r = pd.read_csv(p, index_col='channel').loc[channel]
    if channel == 'err':
        msg += '@here'

    # Create webhook
    webhook = Webhook.partial(r.id, r.token, adapter=RequestsWebhookAdapter())

    # split into strings of max 2000 char for discord
    n = 2000
    out = [(msg[i:i + n]) for i in range(0, len(msg), n)]

    for msg in out:
        webhook.send(msg)


def send_error(msg='', prnt=False):
    import traceback
    err = traceback.format_exc().replace('Traceback (most recent call last):\n', '')

    if not msg == '':
        err = '{}:\n{}'.format(msg, err).replace(':\nNoneType: None', '')

    err = '*------------------*\n{}'.format(err)

    if prnt or not 'linux' in platform:
        print(err)
    else:
        discord(msg=err, channel='err')


# DATABASE
def get_google_sheet():
    import pygsheets
    from google.oauth2 import service_account
    m = SecretsManager('gsheets.json').load
    SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    my_credentials = service_account.Credentials.from_service_account_info(m, scopes=SCOPES)

    # easiest way loading from json file
    # return pygsheets.authorize(service_account_file=p).open('Jambot Settings')
    return pygsheets.authorize(custom_credentials=my_credentials).open('Jambot Settings')


def get_delta(interval=1):
    if interval == 1:
        return delta(hours=1)
    elif interval == 15:
        return delta(minutes=15)


def date_to_dt(d: date) -> dt:
    return dt.combine(d, dt.min.time())


def timenow(interval=1):
    if interval == 1:
        return dt.utcnow().replace(microsecond=0, second=0, minute=0)
    elif interval == 15:
        return round_minutes(dt=dt.utcnow(), resolution=15).replace(microsecond=0, second=0)


def round_minutes(dt, resolution):
    new_minute = (dt.minute // resolution) * resolution
    return dt + delta(minutes=new_minute - dt.minute)


def clean_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return cols if they exist in dataframe"""
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def save_pickle(obj: object, p: Path, name: str):
    """Save object to pickle file"""
    p = p / f'{name}.pkl'
    with open(p, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(p: Path) -> object:
    """Load pickle from file"""
    with open(p, 'rb') as file:
        return pickle.load(file)
