import json
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta as delta
from typing import Tuple
from urllib import parse

import numpy as np
import pandas as pd
import pyodbc
import pypika as pk
import sqlalchemy as sa
from pypika import functions as fn

from jambot import functions as f
from jambot import getlog
from jambot.exchanges.bitmex import Bitmex
from jambot.utils.secrets import SecretsManager

global db

log = getlog(__name__)


def str_conn():
    m = SecretsManager('db.yaml').load
    db_string = ';'.join('{}={}'.format(k, v) for k, v in m.items())
    params = parse.quote_plus(db_string)
    return f'mssql+pyodbc:///?odbc_connect={params}'


def _create_engine():
    """Create sqla engine object
    - sqlalchemy.engine.base.Engine
    - Used in DB class and outside, eg pd.read_sql
    - any errors reading db_creds results in None engine"""

    # connect_args = {'autocommit': True}
    # , isolation_level="AUTOCOMMIT"

    return sa.create_engine(
        str_conn(),
        fast_executemany=True,
        pool_pre_ping=True,
        pool_timeout=5,
        pool_recycle=1700)


class DB(object):
    def __init__(self):
        log.info('Initializing database')
        self.__name__ = 'Jambot Database'
        self.reset(False)

    def close(self):
        if self._engine is None:
            return

        self._engine.raw_connection().close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def engine(self):

        if self._engine is None:
            self._engine = _create_engine()

        if self._engine is None:
            raise Exception('Can\'t connect to database.')

        return self._engine

    @property
    def cursor(self):
        """Raw cursor used for db operations other than refreshing main tables"""
        def _get_cursor():
            return self.engine.raw_connection().cursor()

        try:
            try:
                self._cursor = _get_cursor()
            except (pyodbc.ProgrammingError, pyodbc.OperationalError) as e:
                self.reset()  # retry onece to clear everything then try again
                self._cursor = _get_cursor()
        except Exception as e:
            raise Exception('Couldn\'t create cursor.') from e

        return self._cursor

    def reset(self, warn=True):
        # set engine objects to none to force reset, not ideal
        if warn:
            log.warning('Resetting database.')

        self._engine, self._cursor = None, None

    def get_max_dates(self, interval: int = 1, symbol: str = None) -> pd.DataFrame:
        a = pk.Table('Bitmex')
        q = pk.Query.from_(a) \
            .select('interval', 'symbol', fn.Max(a.timestamp)
                    .as_('timestamp')) \
            .where(a.interval == interval) \
            .groupby(a.symbol, a.interval) \
            .orderby('timestamp')

        # much faster to only check one symbol if thats all needed
        if not symbol is None:
            q = q.where(a.symbol == symbol)

        return pd \
            .read_sql_query(
                sql=q.get_sql(),
                con=self.engine,
                parse_dates=['timestamp']) \
            .assign(interval=lambda x: x.interval.astype(int))

    def update_all_symbols(
            self,
            exch: 'Bitmex' = None,
            interval: int = 1,
            symbol: str = None) -> None:

        lst = []
        if exch is None:
            exch = Bitmex.default(test=False, refresh=False)

        # loop query result, add all to dict with maxtime as KEY, symbols as LIST
        m = defaultdict(list)
        df_max = self.get_max_dates(interval=interval, symbol=symbol)

        # get grouped list of symbols by their max timestamp in db
        for _, row in df_max.iterrows():
            m[row.timestamp].append(row.symbol)

        # loop dict and call bitmex for each list of syms in maxdate
        for maxdate, symbols in m.items():
            starttime = maxdate + f.get_offset(interval)
            if starttime < f.timenow(interval):
                fltr = json.dumps(dict(symbol=symbols))  # filter symbols needed

                df_cdls = exch \
                    .get_candles(
                        starttime=starttime,
                        fltr=fltr,
                        include_partial=False,
                        interval=interval)

                lst.append(df_cdls)

        nrows = 0
        if lst:
            df = pd.concat(lst)
            nrows = df.shape[0]
            df.to_sql(name='Bitmex', con=self.engine, if_exists='append', index=False)

        log.info(f'Imported [{nrows}] rows to db.')

    def get_df(
            self,
            symbol: str,
            period: int = 300,
            startdate: dt = None,
            enddate: dt = None,
            daterange: Tuple[dt] = None,
            interval: int = 1,
            offset: int = -15):

        if startdate is None:
            startdate = f.timenow(interval=interval) + delta(hours=abs(period) * -1)
        else:
            if enddate is None and not daterange is None:
                enddate = startdate + delta(days=daterange)

            startdate += delta(days=offset)

        a = pk.Table('Bitmex')
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        dtypes = {c: np.float32 for c in cols[1:]}
        dtypes['volume'] = pd.Int64Dtype()

        q = pk.Query.from_(a) \
            .select(*cols) \
            .where(a.interval == interval) \
            .where(a.timestamp >= startdate) \
            .orderby('symbol', 'timestamp')

        if not symbol is None:
            q = q.where(a.symbol == symbol)

        if not enddate is None:
            q = q.where(a.timestamp <= enddate)

        return pd.read_sql_query(sql=q.get_sql(), con=self.engine, parse_dates=['timestamp']) \
            .set_index('timestamp', drop=True) \
            .astype(dtypes)


db = DB()


def load_from_start(symbol: str, interval: int = 15):
    """Update database from scratch per symbol/interval

    Parameters
    ----------
    symbol : str
    interval : int, optional
        (eg 1 or 15), default 15
    """
    maxdate = dt(2015, 12, 1)
    i = 1
    exch = Bitmex.default()

    while maxdate < dt.utcnow() + delta(hours=-2):
        df_max = db.get_max_dates(interval=interval, symbol=symbol)

        if len(df_max) > 0:
            maxdate = df_max.timestamp[0].value

        df = exch.get_candles(interval=interval, symbol=symbol, starttime=maxdate)

        df.to_sql(name='Bitmex', con=db.engine, if_exists='append', index=False)

        log.info(f'{i}: {df.timestamp.min()} - {df.timestamp.max()}')
        i += 1
