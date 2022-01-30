from datetime import datetime as dt
from datetime import timedelta as delta
from typing import *
from urllib import parse

import pandas as pd
import pyodbc
import pypika as pk
import sqlalchemy as sa
from pypika import MSSQLQuery as Query
from pypika.queries import QueryBuilder
from sqlalchemy.orm import Session, sessionmaker

from jambot import getlog
from jambot.exchanges.bitmex import Bitmex
from jgutils.secrets import SecretsManager

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
    - any errors reading db_creds results in None engine
    """

    return sa.create_engine(
        str_conn(),
        fast_executemany=True,
        pool_pre_ping=True,
        pool_timeout=5,
        pool_recycle=1700)


class DB(object):

    def __init__(self):
        self.__name__ = 'Jambot Database'
        log.debug(f'Initializing database: {self.__name__}')
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

    @property
    def session(self) -> 'Session':
        if self._session is None:
            self._session = sessionmaker(bind=self.engine)()

        return self._session

    def reset(self, warn=True):
        # set engine objects to none to force reset, not ideal
        if warn:
            log.warning('Resetting database.')

        self._engine, self._cursor, self._session = None, None, None

    def read_sql(self, sql: Union[str, QueryBuilder], **kw) -> pd.DataFrame:
        """Get sql query from db

        Parameters
        ----------
        sql : Union[str, QueryBuilder]
            sql string or pk.Query

        Returns
        -------
        pd.DataFrame
        """
        sql = sql.get_sql() if isinstance(sql, QueryBuilder) else sql
        return pd.read_sql_query(sql=sql, con=self.engine, **kw)

    def join_funding(self, df: pd.DataFrame, df_fund: pd.DataFrame = None, **kw) -> pd.DataFrame:
        """Merge funding rate to ohlc data
        - backfill funding rate
        - NOTE not used, just getting funding from df with join then backfill

        Parameters
        ----------
        df : pd.DataFrame
            df_ohlc
        df_fund : pd.DataFrame
            funding rate data

        Returns
        -------
        pd.DataFrame
            df with funding rate merged
        """
        if df_fund is None:
            df_fund = self.get_funding(**kw)

        return pd.merge_asof(
            left=df,
            right=df_fund,
            left_index=True,
            right_index=True,
            direction='forward',
            allow_exact_matches=True)

    def import_df(
            self,
            df: pd.DataFrame,
            imptable: str,
            impfunc: str,
            prnt: bool = False,
            index: bool = False,
            if_exists: str = 'append',
            import_name: str = None) -> Union[int, None]:

        rowsadded = 0
        if df is None or len(df) == 0:
            if imptable == 'temp_import' and not import_name is None:
                imptable = import_name
            log.warning(f'No rows to import to: {imptable}')
            return

        df.to_sql(
            name=imptable,
            con=self.engine,
            if_exists=if_exists,
            index=index)

        cursor = self.cursor
        rowsadded = cursor.execute(impfunc).rowcount
        cursor.commit()

        import_name = import_name or imptable
        msg = f'{import_name}: {rowsadded}'
        if prnt:
            log.info(msg)

        return rowsadded

    def insert_update(
            self,
            a: str,
            df: pd.DataFrame,
            join_cols: List[str],
            b: str = 'temp_import',
            **kw) -> int:
        """Insert values from df into temp update table b and merge to a

        Parameters
        ----------
        a : str
            insert into table
        b : str
            select from table (temp table)
        join_cols : str
            colums to join a/b on
        df : pd.DataFrame

        Returns
        -------
        int
            rows added
        """
        if b == 'temp_import':
            kw['if_exists'] = 'replace'

        imptable = b

        # drop duplicates
        if not df is None and len(df) > 0:

            # sometimes df will have been converted to lower cols
            join_cols_lower = [c.lower() for c in join_cols]
            subset = join_cols if not all(c in df.columns for c in join_cols_lower) else join_cols_lower

            df = df \
                .drop_duplicates(subset=subset, keep='first')

            a, b = pk.Tables(a, b)
            cols = df.columns

            # this builds an import function from scratch, replaces stored proceedures
            q = Query.into(a) \
                .columns(*cols) \
                .from_(b) \
                .left_join(a).on_field(*join_cols) \
                .select(*cols) \
                .where(a.field(join_cols[0]).isnull())

            print(q.get_sql())
        else:
            q = ''

        rowsadded = self.import_df(df=df, imptable=imptable, impfunc=str(q), import_name=a, **kw)
        # self.cursor.execute(f'TRUNCATE TABLE {b};')
        self.cursor.execute(f'DROP TABLE {b};')
        self.cursor.commit()

        msg = f'{a}: {rowsadded}'
        log.info(msg)

        return rowsadded


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
    exch = Bitmex.default(refresh=False)

    while maxdate < dt.utcnow() + delta(hours=-2):
        df_max = db.get_max_dates(interval=interval, symbol=symbol)

        if len(df_max) > 0:
            maxdate = df_max.timestamp[0].value

        df = exch.get_candles(interval=interval, symbol=symbol, starttime=maxdate)

        df.to_sql(name='Bitmex', con=db.engine, if_exists='append', index=False)

        log.info(f'{i}: {df.timestamp.min()} - {df.timestamp.max()}')
        i += 1
