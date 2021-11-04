import copy
from abc import ABCMeta, abstractproperty
from datetime import datetime as dt
from datetime import timedelta as delta
from typing import *

import numpy as np
import pandas as pd
import pypika as pk
from pypika import Query
from pypika import functions as fn
from pypika.terms import Criterion

from jambot import SYMBOL
from jambot import comm as cm
from jambot import functions as f
from jambot import getlog
from jambot.database import db
from jambot.exchanges.exchange import SwaggerExchange

log = getlog(__name__)


class Table(object, metaclass=ABCMeta):
    """Class to represent database table"""
    name = abstractproperty()
    cols = abstractproperty()
    dtypes = None

    # store exchange name as tinyint
    exch_keys = dict(
        bitmex=1,
        bybit=2)
    exch_keys_inv = f.inverse(exch_keys)

    def __init__(self):
        self.a = pk.Table(self.name)

    def get_query(
            self,
            q: Query = None,
            cols: Union[bool, list] = None,
            conds: List[pk.Criterion] = None,
            **kw) -> Query:
        """Create pk.Query, add conditions etc

        Parameters
        ----------
        q : Query, optional
            default None
        cols : Union[bool, list], optional
            table cols to query, by default self.cols
        conds : List[pk.Criterion], optional
            criterion to filter, by default None

        Returns
        -------
        Query
        """

        q = q or Query.from_(self.a)

        if not cols is False:
            cols = cols or self.cols
            q = q.select(*cols)

        if conds:
            q = q.where(Criterion.all(conds))

        return q

    def process_df(self, df: pd.DataFrame, **kw) -> pd.DataFrame:
        return df

    def get_df(self, prnt: bool = False, **kw) -> pd.DataFrame:
        """Get df from database

        Parameters
        ----------
        prnt : bool, optional
            print query, by default False
        **kw : dict
            args passed to get_query

        Returns
        -------
        pd.DataFrame
        """
        q = self.get_query(**kw)

        if prnt:
            log.info(q)

        df = db.read_sql(sql=q)

        if not self.dtypes is None:
            df = df.astype(self.dtypes)

        df = df.pipe(self.process_df, **kw)

        if hasattr(self, 'idx_cols'):
            df = df.set_index(self.idx_cols)

        return df

    def _get_max_dates(
            self,
            cols: List[str],
            symbols: List[str] = None,
            conds: List[pk.Criterion] = None) -> pd.DataFrame:
        """Get df of max dates per exchange/symbol for table

        Parameters
        ----------
        cols : List[str]
        symbols : List[str], optional
            default None (all symbols in table)
        conds : List[pk.Criterion], optional
            default None

        Returns
        -------
        pd.DataFrame
        """

        q = Query.from_(self.a) \
            .select(*cols, fn.Max(self.a.timestamp).as_('timestamp')) \
            .groupby(*cols) \
            .orderby('timestamp')

        conds = conds or []
        if not symbols is None:
            conds.append(self.a.symbol.isin(f.as_list(symbols)))

        if conds:
            q = q.where(Criterion.all(f.as_list(conds)))

        df = db.read_sql(sql=q)

        if 'exchange' in cols:
            df = df.assign(exchange=lambda x: x.exchange.apply(lambda x: self.exch_keys_inv[x]))

        return df

    def update_from_exch(
            self,
            exchs: Union[SwaggerExchange, List[SwaggerExchange]],
            symbols: Union[str, List[str]] = None,
            test: bool = False,
            **kw) -> None:
        """Update table data from exchange based on max dates in table

        Parameters
        ----------
        exchs : Union[SwaggerExchange, List[SwaggerExchange]]
            single or multiple exchanges to get data from
        symbols : Union[str, List[str]], optional
            default exchange's default symbol
        test : bool, optional
            only print output, dont import, default False
        """

        # use exch default symbols if not given
        exchs = f.as_list(exchs)
        symbols = symbols or [exch.default_symbol for exch in exchs]

        # convert exchanges to dict for matching by num
        exchs = {exch.exch_name: exch for exch in exchs}
        dfs = []

        # get max symbol grouped by timestamp/interval/exch
        df_max = self.get_max_dates(symbols=symbols, **kw)

        for (exch_name, timestamp), df in df_max.groupby(['exchange', 'timestamp']):
            # get exchange obj from exch_num
            df = self._get_exch_data(
                exch=exchs[exch_name],
                timestamp=timestamp,
                symbols=df['symbol'].tolist(), **kw)

            if not df is None:
                dfs.append(df)

        nrows = 0
        nsymbols = 0

        if dfs:
            df = pd.concat(dfs)
            nrows = df.shape[0]
            nsymbols = df.symbol.nunique()

            if not test:
                df.to_sql(name=self.name, con=db.engine, if_exists='append', index=False)

        msg = f'Imported [{nrows}] row(s) for [{nsymbols}] symbol(s)'
        log.info(msg)

        if self.name == 'funding':
            cm.discord(f'Funding: {msg}', channel='test')


class Tickers(Table):
    name = 'bitmex'
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    idx_cols = 'timestamp'
    dtypes = {c: np.float32 for c in cols[1:]}
    dtypes['volume'] = pd.Int64Dtype()

    def __init__(self):
        super().__init__()

    def get_query(
            self,
            exch_name: str = 'bitmex',
            symbol: str = SYMBOL,
            period: int = 300,
            startdate: dt = None,
            enddate: dt = None,
            interval: int = 15,
            funding: bool = False,
            **kw) -> pd.DataFrame:

        # TODO allow symbols = dict(bitmex=['XBTUSD'])
        # this could actually just be a "build conds"

        if startdate is None:
            startdate = f.inter_now(interval=interval) + delta(hours=abs(period) * -1)

            # add extra offset for building signals (eg ema_200)
            offset = {1: 16, 15: 4}.get(interval)
            startdate += delta(days=-offset)

        a = self.a
        q = Query.from_(a)
        cols = copy.copy(self.cols)

        # add funding column
        if funding:
            b = pk.Table('funding')
            q = q \
                .left_join(b) \
                .on_field('exchange', 'symbol', 'timestamp')

            cols.append(b.funding_rate)

        conds = [
            a.interval == interval,
            a.timestamp >= startdate,
            a.exchange == self.exch_keys[exch_name]
        ]

        if not symbol is None:
            conds.append(a.symbol == symbol)

        if not enddate is None:
            conds.append(a.timestamp <= enddate)

        return super().get_query(q=q, conds=conds, cols=cols)

    def process_df(self, df: pd.DataFrame, funding_exch: SwaggerExchange = None, **kw) -> pd.DataFrame:
        """Add funding rate

        Parameters
        ----------
        df : pd.DataFrame
        funding_exch : SwaggerExchange, optional
            used to fill last n rows with "next_funding", default None

        Returns
        -------
        pd.DataFrame
        """

        # backfill funding rate
        if 'funding_rate' in df.columns:
            df = df.assign(funding_rate=lambda x: x.funding_rate.backfill().astype(np.float32))

            if funding_exch:
                # NOTE XBTUSD will have to change
                df = df.assign(
                    funding_rate=lambda x: x.funding_rate.fillna(funding_exch.next_funding('XBTUSD')))

        return df

    def get_max_dates(self, interval: int = 15, symbols: List[str] = None) -> pd.DataFrame:

        a = self.a
        cols = ['exchange', 'interval', 'symbol']
        conds = [a.interval == interval]

        return self._get_max_dates(cols=cols, symbols=symbols, conds=conds) \
            .assign(interval=lambda x: x.interval.astype(int))

    def _get_exch_data(
            self,
            exch: SwaggerExchange,
            timestamp: dt,
            symbols: str,
            interval: int = 15) -> pd.DataFrame:
        """Get ticker data from exchange

        Parameters
        ----------
        exch : SwaggerExchange
        timestamp : dt
        symbols : str
        interval : int, optional
            default 15

        Returns
        -------
        pd.DataFrame
        """
        starttime = timestamp + f.inter_offset(interval)

        if starttime < f.inter_now(interval):
            return exch \
                .get_candles(
                    starttime=starttime,
                    symbol=symbols,
                    include_partial=False,
                    interval=interval) \
                .assign(exchange=self.exch_keys[exch.exch_name])


class Funding(Table):
    name = 'funding'
    cols = ['exchange', 'symbol', 'timestamp', 'funding_rate']
    idx_cols = 'timestamp'
    dtypes = dict(funding_rate=np.float32)

    def __init__(self):
        super().__init__()

    def get_query(self, **kw) -> pd.DataFrame:

        a = self.a
        cols = [a.timestamp, a.funding_rate]

        return super().get_query(cols=cols)

    def get_max_dates(self, symbols: List[str] = None) -> pd.DataFrame:

        a = self.a
        cols = ['exchange', 'symbol']

        return self._get_max_dates(cols=cols, symbols=symbols)

    def _get_exch_data(
            self,
            exch: SwaggerExchange,
            timestamp: dt,
            symbols: str,
            **kw) -> pd.DataFrame:
        """Get funding data
        - TODO will need to add Bybit.get_funding_history soon

        Parameters
        ----------
        exch : SwaggerExchange
        timestamp : dt
        symbols : str

        Returns
        -------
        pd.DataFrame
        """

        # next funding to get always 8hrs after last
        starttime = timestamp + delta(hours=8)

        if starttime <= dt.utcnow():
            return exch \
                .get_funding_history(
                    symbol=symbols,
                    starttime=starttime) \
                .assign(exchange=self.exch_keys[exch.exch_name]) \
                .reset_index(drop=False)


class ApiKeys(Table):
    name = 'apikeys'
    cols = ['exchange', 'user', 'key', 'secret']
    idx_cols = ['exchange', 'user']
