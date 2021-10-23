
import json
import time
import warnings
from datetime import datetime as dt
from datetime import timedelta as delta
from datetime import timezone as tz
from typing import *

import pandas as pd
from bitmex import bitmex
from BitMEXAPIKeyAuthenticator import APIKeyAuthenticator

from jambot import SYMBOL
from jambot import functions as f
from jambot import getlog
from jambot.exchanges.exchange import SwaggerExchange
from jambot.tradesys.orders import ExchOrder

log = getlog(__name__)
warnings.filterwarnings('ignore', category=Warning, message='.*format is not registered')


class Bitmex(SwaggerExchange):
    div = 1e8
    default_symbol = SYMBOL
    conv_symbol = dict(BTCUSD=default_symbol)
    wallet_keys = dict(
        avail_margin='excessMargin',
        total_balance_margin='marginBalance',
        total_balance_wallet='walletBalance',
        unrealized_pnl='unrealisedPnl',
        prev_pnl='prevRealisedPnl',
        value='maintMargin')

    other_keys = dict(
        last_price='lastPrice',
        cur_qty='currentQty',
        err_text='text')

    # value = "current value of position"
    pos_keys = dict(
        sym_short='underlying',
        qty='currentQty',
        entry_price='avgEntryPrice',
        last_price='lastPrice',
        u_pnl='unrealisedPnl',
        pnl_pct='unrealisedPnlPcnt',
        roe_pct='unrealisedRoePcnt',
        value='maintMargin'
    )

    order_keys = ExchOrder._m_conv.get('bitmex')
    api_host = 'https://www.bitmex.com'
    api_host_test = 'https://testnet.bitmex.com'
    api_spec = '/api/explorer/swagger.json'

    order_params = dict(
        submit=dict(
            func='Order_newBulk',
            order_kw='orders',
            keys=('symbol', 'ordType', 'orderQty', 'clOrdID', 'execInst', 'price', 'stopPx')),
        amend=dict(
            func='Order_amendBulk',
            order_kw='orders',
            keys=('orderID', 'symbol', 'orderQty', 'price', 'stopPx')),
        cancel=dict(
            func='Order_cancel',
            order_kw=order_keys['order_id'],
            keys=('orderID',)),
        cancel_all=dict(
            func='Order_cancelAll'))

    def __init__(self, user: str, test: bool = False, refresh: bool = False, **kw):
        super().__init__(user=user, test=test, **kw)
        self.partialcandle = None

        f.set_self(vars(), exclude=('refresh',))

        if refresh:
            self.refresh()

    @staticmethod
    def client_cls():
        return bitmex

    @staticmethod
    def client_api_auth():
        return APIKeyAuthenticator

    def _set_positions(self) -> List[dict]:
        positions = self.client.Position.Position_get().result()[0]

        # update position keys for easier access
        for p in positions:
            p |= {k: p[k] / self.div for k in self.wallet_keys.values() if k in p}

        return positions

    def set_leverage(self, symbol: str = None, lev: float = 7.0) -> None:
        symbol = symbol or self.default_symbol
        self.check_request(self.client.Position.Position_updateLeverage(symbol=symbol, leverage=lev))

    def _get_instrument(self, **kw) -> dict:
        return self.client.Instrument.Instrument_get(**kw).response().result[0]

    def get_filled_orders(self, symbol: str = SYMBOL, starttime: dt = None) -> List[ExchOrder]:
        """Get orders filled since last starttime

        - NOTE This refreshes and sets exch orders to recent Filled/PartiallyFilled only

        Parameters
        ----------
        symbol : str, optional
            default XBTUSD
        starttime : dt, optional
            time to filter on, by default None

        Returns
        -------
        List[ExchOrder]
            list of recently filled orders
        """
        if starttime is None:
            starttime = dt.utcnow() + delta(days=-7)

        fltr = dict(ordStatus=['Filled', 'PartiallyFilled'])

        self.set_orders(fltr=fltr, starttime=starttime, reverse=False)
        return self.exch_order_from_raw(order_specs=self._orders, process=False)

    def set_orders(
            self,
            fltr: dict = None,
            count: int = 100,
            starttime: dt = None,
            reverse: bool = True,
            **kw) -> None:
        """Save last count orders from bitmex"""

        if not fltr is None:
            fltr = json.dumps(fltr)

        orders = self.client.Order \
            .Order_getOrders(
                filter=fltr,
                reverse=reverse,
                count=count,
                startTime=starttime) \
            .response().result

        self._orders = self.add_custom_specs(orders)

    def funding_rate(self, symbol=SYMBOL):
        """Get current funding rate from exchange"""
        result = self.client.Instrument.Instrument_get(symbol=symbol).response().result[0]

        rate = result['fundingRate']
        hrs = int((result['fundingTimestamp'] - f.inter_now().replace(tzinfo=tz.utc)).total_seconds() / 3600)

        return rate, hrs

    def _get_total_balance(self) -> dict:
        return self.check_request(self.client.User.User_getMargin(currency='XBt'))

    def _route_order_request(self, action: str, order_specs: List[ExchOrder]):

        # could check/set exchange here

        # convert everything to list of dicts first?
        params = self.order_params.get(action)
        keys = params.get('keys', None)
        order_kw = params.get('order_kw', None)
        func = getattr(self.client.Order, params['func'])

        # filter correct keys for action
        if keys:
            order_specs = [
                {k: v for k, v in s.items() if k in keys and not v is None}
                for s in order_specs]

        if action == 'cancel_all':
            # cancel_all just uses dict(symbol='XBTUSD')
            if isinstance(order_specs, list):
                order_kws = order_specs[0]
            else:
                order_kws = order_specs
        else:
            if action == 'cancel':
                # cancel just needs list of orderIds
                order_specs = f.flatten_list_list([list(s.values()) for s in order_specs])

            order_kws = {order_kw: json.dumps(order_specs)}

        return self.check_request(func(**order_kws))

    def get_partial(self, symbol):
        timediff = 0

        if not self.partialcandle is None:
            timediff = (self.partialcandle.timestamp[0] - f.inter_now()).seconds

        if (timediff > 7200
            or self.partialcandle is None
                or not self.partialcandle.symbol[0] == symbol):
            self.set_partial(symbol=symbol)

        return self.partialcandle

    def set_partial(self, symbol):
        # call only partial candle from exchange, save to self.partialcandle
        starttime = dt.utcnow() + delta(hours=-2)
        self.get_candles(symbol=symbol, starttime=starttime)

    def append_partial(self, df):
        # partial not built to work with multiple symbols, need to add partials to dict
        # Append partialcandle df to df from SQL db

        symbol = df.symbol[0]
        dfpartial = self.get_partial(symbol=symbol)

        return df.append(dfpartial, sort=False).reset_index(drop=True)

    def resample(self, df: pd.DataFrame, include_partial: bool = False, do: bool = True) -> pd.DataFrame:
        """
        Convert 5min candles to 15min
            - need to only include groups of 3 > drop last 1 or 2 rows
            - remove incomplete candles, split into groups first
        """
        if not do:
            return df

        gb = df.groupby('symbol')
        lst = []

        for symbol in gb.groups:
            df = gb.get_group(symbol)

            rs = df \
                .resample('15Min', on='timestamp')

            df = rs \
                .agg(dict(
                    symbol='first',
                    open='first',
                    high='max',
                    low='min',
                    close='last',
                    volume='sum')) \
                .join(rs.timestamp.count().rename('num'))

            if not include_partial:
                df = df.query('num == 3')

            lst.append(df)

        return pd.concat(lst) \
            .reset_index() \
            .drop(columns=['num'])

    def wait_candle_avail(self, interval: int = 1, symbol: str = SYMBOL) -> None:
        """Wait till last period candle is available from exchange
            - rest api latency is ~15s
            - Just for testing, not used for anything

        Parameters
        ----------
        interval : int, optional
            default 1
        """

        # set current lower time bin to check for
        dnow = f.inter_now(interval) - f.inter_offset(interval)

        _time_last = lambda: self.last_candle(interval, symbol=symbol).timestamp.iloc[0]
        time_last = _time_last()

        i = 0
        max_tries = 30

        while time_last < dnow and i < max_tries:
            log.info(dt.utcnow())
            time.sleep(1)
            time_last = _time_last()
            i += 1

    def last_candle(self, interval: int = 1, **kw) -> pd.DataFrame:
        """Return last candle only (for checking most recent avail time)

        Parameters
        ----------
        interval : int, optional
            default 1

        Returns
        -------
        pd.DataFrame
            df of last candle
        """
        count = {1: 1, 15: 5}.get(interval, 1)
        return self.get_candles(interval=interval, reverse=True, count=count, include_partial=False, **kw)

    def get_candles(
            self,
            symbol: str = SYMBOL,
            starttime: dt = None,
            fltr: str = '',
            retain_partial: bool = False,
            include_partial: bool = True,
            interval: int = 1,
            count: int = 1000,
            pages: int = 100,
            reverse: bool = False) -> pd.DataFrame:
        """Get df of OHLCV candles from Bitmex converted to df

        - NOTE bitmex gives closetime, tradingview shows opentime > offset to match open

        Parameters
        ----------
        symbol : str, optional
        starttime : dt, optional
        fltr : str, optional
            any filter which has been json.dumps already, by default ''
        retain_partial : bool, optional
            keep partial candle in return df, by default False
        include_partial : bool, optional
            query include partial candle from bitmex, by default True
        interval : int, optional
        count : int, optional
            max number of candles to include, by default 1000
        pages : int, optional
            by default 100
        reverse : bool, optional
            return candles in reverse (newest first) order, by default False

        Returns
        -------
        pd.DataFrame
            df of OHLCV candles
        """

        if interval == 1:
            binsize = '1h'
            offset = delta(hours=1)
        elif interval == 15:
            binsize = '5m'
            offset = delta(minutes=5)

        if not starttime is None:
            starttime += offset

        resultcount = float('inf')
        start = 0
        page_max = 1000
        lst = []
        cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

        if isinstance(symbol, list) and fltr == '':
            fltr = json.dumps(dict(symbol=symbol))  # filter symbols needed
            symbol = None

        while resultcount >= page_max and start // page_max <= pages:
            request = self.client.Trade.Trade_getBucketed(
                binSize=binsize,
                symbol=symbol,
                startTime=starttime,
                filter=fltr,
                count=count,
                start=start,
                reverse=reverse,
                partial=include_partial,
                columns=json.dumps(cols))

            response = request.response()
            result = response.result
            ratelim_remaining = int(response.metadata.headers['x-ratelimit-remaining'])

            resultcount = len(result)
            lst.extend(result)
            start += page_max

            if ratelim_remaining <= 1:
                log.info('Ratelimit reached. Sleeping 10 seconds.')
                time.sleep(10)

        df = pd.json_normalize(lst) \
            .assign(timestamp=lambda x: x.timestamp.dt.tz_localize(None) + offset * -1) \
            .pipe(self.resample, include_partial=include_partial, do=interval == 15) \
            .assign(interval=interval)[['interval'] + cols]

        if include_partial:
            self.partialcandle = df.tail(1).copy().reset_index(drop=True)  # save last as df

            if not retain_partial:
                df.drop(df.index[-1], inplace=True)

        return df
