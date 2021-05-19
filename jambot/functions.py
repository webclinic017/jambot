# General Functions module - don't rely on any other modules from jambot
import json
import os
from datetime import datetime as dt
from datetime import timedelta as delta
from pathlib import Path
from sys import platform
from time import time

import pandas as pd
import pyodbc
import pypika as pk
import yaml
from dateutil.parser import parse
from pypika import functions as fn

try:
    from IPython.display import display
except ModuleNotFoundError:
    pass

global topfolder
topfolder = Path(__file__).parent


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
    end = time()
    duration = end - start
    if duration < 60:
        ans = '{:.3f}s'.format(duration)
    else:
        mins = duration / 60
        secs = duration % 60
        ans = '{:.0f}m {:.3f}s'.format(mins, secs)

    print(ans)


def line():
    return '\n'


def dline():
    return line() + line()


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


def get_price(pnl, entryprice, side):
    if side == 1:
        return pnl * entryprice + entryprice
    elif side == -1:
        return entryprice / (1 + pnl)


def get_pnl_xbt(contracts=0, entryprice=0, exitprice=0, isaltcoin=False):
    if 0 in (entryprice, exitprice):
        return 0
    elif not isaltcoin:
        return round(contracts * (1 / entryprice - 1 / exitprice), 8)
    elif isaltcoin:
        return round(contracts * (exitprice - entryprice), 8)


def get_pnl(side, entryprice, exitprice):
    if 0 in (entryprice, exitprice):
        return 0
    elif side == 1:
        return round((exitprice - entryprice) / entryprice, 4)
    else:
        return round((entryprice - exitprice) / exitprice, 4)


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
            'execInst', 'ordStatus', 'contracts', 'name', 'manual', 'orderID')
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
    p = topfolder / 'data/ApiKeys/binance.yaml'
    with open(p) as file:
        m = yaml.full_load(file)
    return m


def discord(msg, channel='jambot'):
    import discord
    import requests
    from discord import File, RequestsWebhookAdapter, Webhook

    p = Path(topfolder) / 'data/ApiKeys/discord.csv'
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
    p = topfolder / 'data/ApiKeys/gsheets.json'
    return pygsheets.authorize(service_account_file=p).open('Jambot Settings')


def get_delta(interval=1):
    if interval == 1:
        return delta(hours=1)
    elif interval == 15:
        return delta(minutes=15)


def timenow(interval=1):
    if interval == 1:
        return dt.utcnow().replace(microsecond=0, second=0, minute=0)
    elif interval == 15:
        return round_minutes(dt=dt.utcnow(), resolution=15).replace(microsecond=0, second=0)


def round_minutes(dt, resolution):
    new_minute = (dt.minute // resolution) * resolution
    return dt + delta(minutes=new_minute - dt.minute)


class Switch:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False  # Allows a traceback to occur

    def __call__(self, *values):
        return self.value in values
