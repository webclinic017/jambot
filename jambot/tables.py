import copy
from abc import ABCMeta, abstractproperty
from datetime import datetime as dt
from datetime import timedelta as delta
from typing import *

import numpy as np
import pandas as pd
import pypika as pk
from pypika import MSSQLQuery as Query
from pypika import functions as fn
from pypika.terms import Criterion

from jambot import SYMBOL
from jambot import comm as cm
from jambot import functions as f
from jambot import getlog
from jambot.database import db
from jambot.exchanges.exchange import SwaggerExchange
from jgutils import functions as jf

if TYPE_CHECKING:
    from pypika.queries import QueryBuilder

log = getlog(__name__)

# TODO set up funding rate data for all symbols


class Table(object, metaclass=ABCMeta):
    """Class to represent database table"""
    name = abstractproperty()
    cols = abstractproperty()
    dtypes = {}  # type: Dict[str, Any]
    idx_cols = abstractproperty()  # type Union[List[str], str]

    # store exchange name as tinyint
    exch_keys = dict(bitmex=1, bybit=2, binance=3)
    exch_keys_inv = jf.inverse(exch_keys)

    def __init__(self):
        self.a = pk.Table(self.name)

    def get_query(
            self,
            q: Optional['QueryBuilder'] = None,
            cols: Optional[Union[bool, list]] = None,
            conds: List[pk.Criterion] = None,
            **kw) -> 'QueryBuilder':
        """Create pk.Query, add conditions etc

        Parameters
        ----------
        q : QueryBuilder, optional
            default None
        cols : Union[bool, list], optional
            table cols to query, by default self.cols
        conds : List[pk.Criterion], optional
            criterion to filter, by default None

        Returns
        -------
        QueryBuilder
        """

        q = q or Query.from_(self.a)  # type: QueryBuilder

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
            conds.append(self.a.symbol.isin(jf.as_list(symbols)))

        if conds:
            q = q.where(Criterion.all(jf.as_list(conds)))

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
        _exchs = jf.as_list(exchs)
        symbols = symbols or [exch.default_symbol for exch in _exchs]

        # convert exchanges to dict for matching by num
        exchs = {exch.exch_name: exch for exch in _exchs}
        dfs = []

        # get max symbol grouped by timestamp/interval/exch
        df_max = self.get_max_dates(symbols=symbols, exchs=_exchs, **kw)

        for (exch_name, timestamp), df in df_max.groupby(['exchange', 'timestamp']):
            # get exchange obj from exch_num, get candle data from exch
            df = self._get_exch_data(
                exch=exchs[exch_name],
                timestamp=timestamp,
                symbols=df['symbol'].tolist(), **kw)

            if not df is None:
                dfs.append(df)

        nrows = 0
        nsymbols = 0

        if dfs:
            df = pd.concat(dfs)  # type: pd.DataFrame
            nrows = df.shape[0]
            nsymbols = df.symbol.nunique()

            if not test:
                self.load_to_db(df=df)

        msg = f'{self.name}: Imported [{nrows:,.0f}] row(s) for [{nsymbols}] symbol(s)'
        log.info(msg)

        if self.name == 'funding':
            cm.discord(f'Funding: {msg}', channel='test')

    def load_to_db(self, df: pd.DataFrame, **kw) -> None:
        """Load data to database

        Parameters
        ----------
        df : pd.DataFrame
        """
        # check if df index cols match self.idx_cols
        index = True if sorted(df.index.names) == sorted(self.idx_cols) else False

        log.info(f'Loading [{len(df):,.0f}] rows to db.')

        df.to_sql(name=self.name, con=db.engine, if_exists='append', index=index)


class Tickers(Table):
    name = 'bitmex'
    idx_cols = ['symbol', 'timestamp']
    # idx_cols = ['timestamp']
    cols = idx_cols + ['open', 'high', 'low', 'close', 'volume']
    dtypes = {c: np.float32 for c in cols[len(idx_cols):]}
    dtypes['volume'] = pd.Int64Dtype()
    dtypes['symbol'] = pd.CategoricalDtype()

    def __init__(self):
        super().__init__()

    def get_query(
            self,
            exch_name: str = 'bitmex',
            symbols: Union[str, List[str]] = SYMBOL,
            period: int = 300,
            startdate: dt = None,
            enddate: dt = None,
            interval: int = 15,
            # funding: bool = False,
            **kw) -> pd.DataFrame:
        """Get df of OHLCV + funding_rate
        - funding_exch passed to process_df to fill last funding row
        - TODO add SYMBOL to index and fix everything everywhere
            - eg allow symbols = dict(bitmex=['XBTUSD'])

        Parameters
        ----------
        exch_name : str, optional
            default 'bitmex'
        symbols : Union[str, List[str]], optional
            default SYMBOL
        period : int, optional
            last n periods, default 300 (only used if startdate not given)
        startdate : dt, optional
            default None
        enddate : dt, optional
            default None
        interval : int, optional
            default 15

        Returns
        -------
        pd.DataFrame
        """
        # this could actually just be a "build conds"

        # FIXME temp solution use binance for other symbols
        if not symbols == 'XBTUSD' and exch_name == 'bitmex':
            exch_name = 'binance'

        if startdate is None:
            startdate = f.inter_now(interval=interval) + delta(hours=abs(period) * -1)

            # add extra offset for building signals (eg ema_200)
            offset = {1: 16, 15: 4}[interval]
            startdate += delta(days=-offset)

        a = self.a
        q = Query.from_(a)
        cols = copy.copy(self.cols)

        # add funding column, will be NaN if not present
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

        if not symbols is None:
            conds.append(a.symbol.isin(jf.as_list(symbols)))
            # conds.append(a.symbol == symbols)

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
        return df.assign(funding_rate=0)

        # backfill funding rate
        if 'funding_rate' in df.columns:
            df = df.assign(funding_rate=lambda x: x.funding_rate.backfill().astype(np.float32))

            if funding_exch:
                # FIXME XBTUSD will have to change
                # TODO grouby symbol before adding latest funding
                df = df.assign(
                    funding_rate=lambda x: x.funding_rate.fillna(funding_exch.next_funding(SYMBOL)))

        return df

    def get_max_dates(
            self,
            interval: int = 15,
            symbols: List[str] = None,
            exchs: Union[SwaggerExchange, List[SwaggerExchange]] = None) -> pd.DataFrame:

        a = self.a
        cols = ['exchange', 'interval', 'symbol']
        conds = [a.interval == interval]

        # filter max_dates to specific exchanges
        if not exchs is None:
            exch_names = [self.exch_keys[exch.exch_name] for exch in jf.as_list(exchs)]
            conds.append(self.a.exchange.isin(exch_names))

        # see if ohlc data exists for symbol/exchange
        df = self._get_max_dates(cols=cols, symbols=symbols, conds=conds) \
            .assign(interval=lambda x: x.interval.astype(int))

        # create max date data from earliest record of symbol on exchange
        if len(df) == 0:
            if exchs is None:
                raise RuntimeError(f'Must provide exchange to get max dates for missing symbol(s): {symbols}')

            data = []
            for exch in jf.as_list(exchs):
                for symbol in jf.as_list(symbols):
                    m = dict(
                        exchange=exch.exch_name,
                        interval=interval,
                        symbol=symbol,
                        timestamp=exch.get_earliest_candle_date(symbol=symbol, interval=interval))
                    data.append(m)

            df = pd.DataFrame(data=data)

        return df

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

    def get_max_dates(self, symbols: List[str] = None, **kw) -> pd.DataFrame:

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


class Predictions(Table):
    """Historical model predictions"""
    name = 'predictions'
    idx_cols = ['interval', 'symbol', 'timestamp']
    cols = idx_cols + ['signal']

    def load_to_db(
            self,
            df: pd.DataFrame,
            # n_hours: int = 24,
            interval: int = 15,
            symbol: str = SYMBOL,
            test: bool = False) -> None:
        """Check max preds date in db and load new signals

        Parameters
        ----------
        df : pd.DataFrame
            df_ohlc with signal col
        interval : int, optional
            default 15
        symbol : str, optional
            default 'XBTUSD'
        """

        df_max = self.get_max_dates()

        # filter greater than max date
        if len(df_max) > 0:
            df = df.loc[df.index > df_max.timestamp.max()]

        # trim to last n periods
        # d_min = df.index.max() + delta(hours=-n_hours)

        # drop zeroes
        # drop ohlc
        # assign signal, interval
        df = df[['signal']] \
            .pipe(lambda df: df[df.signal != 0]) \
            .assign(interval=interval, symbol=symbol) \
            .reset_index(drop=False) \
            .dropna()

        if test:
            from jambot import display
            display(df)
            return

        try:
            super().load_to_db(df=df)
            log.info(f'loaded [{len(df)}] predictions to db.')
        except Exception as e:
            # TODO send discord error
            log.warning('Failed to load preds to db.')

    def process_df(self, df: pd.DataFrame, merge: bool = False, df_ohlc: pd.DataFrame = None, **kw) -> pd.DataFrame:

        # if merge, merge to full df_ohlc
        if merge:
            pass

        return df

    def get_max_dates(self, interval: int = 15, symbols: List[str] = None) -> pd.DataFrame:

        a = self.a
        cols = ['interval', 'symbol']
        conds = [a.interval == interval]

        return self._get_max_dates(cols=cols, symbols=symbols, conds=conds) \
            .assign(interval=lambda x: x.interval.astype(int))


class Symbols(Table):
    name = 'symbols'
    idx_cols = ['exchange', 'symbol']
    cols = idx_cols + ['base_currency', 'quote_currency', 'is_inverse', 'lot_size', 'tick_size', 'prec']

    def update_from_exch(
            self,
            exchs: Union[SwaggerExchange, List[SwaggerExchange]],
            **kw) -> pd.DataFrame:
        """Update table data from exchange based on max dates in table

        Parameters
        ----------
        exchs : Union[SwaggerExchange, List[SwaggerExchange]]
            single or multiple exchanges to get data from
        """

        # convert exchanges to dict for matching by num
        dfs = []

        for exch in jf.as_list(exchs):
            # get exchange obj from exch_num, get candle data from exch
            df = exch.instrument_data() \
                .reset_index(drop=False)

            if not df is None:
                dfs.append(df)

        df = pd.concat(dfs)  # type: pd.DataFrame

        # drop everything per exchange before loading new
        names = [exch.exch_name for exch in jf.as_list(exchs)]
        a = pk.Table(self.name)
        q = Query.from_(a).delete().where(a.exchange.isin(names))
        cursor = db.cursor
        cursor.execute(q.get_sql())
        cursor.commit()
        self.load_to_db(df=df)

        return df
