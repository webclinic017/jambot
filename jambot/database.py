import json
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta as delta
from pathlib import Path
from urllib import parse as prse

import pandas as pd
import pypika as pk
import sqlalchemy as sa
import yaml
from pypika import Case, Criterion
from pypika import CustomFunction as cf
from pypika import Order, Query
from pypika import functions as fn

from jambot import functions as f
from jambot import getlog
from jambot.exchanges.bitmex import Bitmex
from jambot.utils.secrets import SecretsManager

global db

log = getlog(__name__)


def strConn():
    m = SecretsManager('db.yaml').load
    return ';'.join('{}={}'.format(k, v) for k, v in m.items())


def engine():
    params = prse.quote_plus(strConn())
    return sa.create_engine(f'mssql+pyodbc:///?odbc_connect={params}', fast_executemany=True)


class DB(object):
    def __init__(self):
        log.info('Initializing database')
        self.__name__ = 'Jambot Database'
        self.df_unit = None
        self.conn = engine()
        self.conn.raw_connection().autocommit = True  # doesn't seem to work rn
        self.cursor = self.conn.raw_connection().cursor()

    def close(self):
        try:
            self.cursor.close()
        except:
            try:
                self.conn.raw_connection().close()
            except:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_max_dates(self, interval=1):
        a = pk.Table('Bitmex')
        q = pk.Query.from_(a) \
            .select('Interval', 'Symbol', fn.Max(a.Timestamp)
                    .as_('Timestamp')) \
            .where(a.Interval == interval) \
            .groupby(a.Symbol, a.Interval) \
            .orderby('Timestamp')

        return pd \
            .read_sql_query(
                sql=q.get_sql(),
                con=self.conn,
                parse_dates=['Timestamp']) \
            .assign(Interval=lambda x: x.Interval.astype(int))

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
        df_max = self.get_max_dates(interval=interval)

        # filter single symbol to update (eg XTBUSD on 15 min)
        if not symbol is None:
            df_max = df_max.query('Symbol == @symbol')

        for _, row in df_max.iterrows():
            m[row.Timestamp].append(row.Symbol)

        # loop dict and call bitmex for each list of syms in maxdate
        for maxdate in m.keys():
            starttime = maxdate + f.get_delta(interval)
            if starttime < f.timenow(interval):
                fltr = json.dumps(dict(symbol=m[maxdate]))  # filter symbols needed

                dfcandles = exch \
                    .get_candles(
                        starttime=starttime,
                        fltr=fltr,
                        includepartial=False,
                        interval=interval)

                lst.append(dfcandles)

        if lst:
            df = pd.concat(lst)  # maybe remove duplicates
            df.to_sql(name='Bitmex', con=self.conn, if_exists='append', index=False)

    def get_dataframe(
            self,
            symbol=None,
            period=300,
            startdate=None,
            enddate=None,
            daterange=None,
            interval=1,
            offset=-15):

        if startdate is None:
            startdate = f.timenow(interval=interval) + delta(hours=abs(period) * -1)
        else:
            if enddate is None and not daterange is None:
                enddate = startdate + delta(days=daterange)

            startdate += delta(days=offset)

        a = pk.Table('Bitmex')
        cols = ['Symbol', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'VolBTC']

        q = pk.Query.from_(a) \
            .select(*cols) \
            .where(a.Interval == interval) \
            .where(a.Timestamp >= startdate) \
            .orderby('Symbol', 'Timestamp')

        if not symbol is None:
            q = q.where(a.Symbol == symbol)

        if not enddate is None:
            q = q.where(a.Timestamp <= enddate)

        df = pd.read_sql_query(sql=q.get_sql(), con=self.conn, parse_dates=['Timestamp']) \
            .set_index('Timestamp', drop=False)

        return df


db = DB()
