
import json
import time
import warnings
from datetime import datetime as dt
from datetime import timedelta as delta
from datetime import timezone as tz
from typing import *

import numpy as np
import pandas as pd
from bitmex import bitmex
from BitMEXAPIKeyAuthenticator import APIKeyAuthenticator
from bravado.exception import HTTPBadRequest

from jambot import SYMBOL
from jambot import functions as f
from jambot import getlog
from jambot.exchanges.exchange import SwaggerAPIException, SwaggerExchange
from jambot.tradesys.orders import ExchOrder
from jgutils import functions as jf

if TYPE_CHECKING:
    from bravado.http_future import HttpFuture

log = getlog(__name__)
warnings.filterwarnings('ignore', category=Warning, message='.*format is not registered')


class BitmexAuth(APIKeyAuthenticator):
    """
    Wrap BitmexAPIKeyAuthenticator to add longer expiry timeout
    """

    def apply(self, r):
        # 20s grace period in case of clock skew
        timeout = 20
        expires = int(round(time.time()) + timeout)
        r.headers['api-expires'] = str(expires)
        r.headers['api-key'] = self.api_key
        prepared = r.prepare()
        body = prepared.body or ''
        url = prepared.path_url
        r.headers['api-signature'] = self.generate_signature(self.api_secret, r.method, url, expires, body)
        return r


class BitmexAPIException(SwaggerAPIException):
    def __init__(
            self,
            e: HTTPBadRequest,
            request: 'HttpFuture',
            fail_msg: str = None) -> None:
        """Raise exception on Bitmex invalid api request

        Parameters
        ----------
        e: HTTPBadRequest
            Exception raised by bitmex API
        request : HttpFuture
            original request to get operation name
        fail_msg : str, optional
            custom additional err info message, by default None
        request_kw : dict, optional
            kws passed to request, by default None
        """
        data = request.future.request.data  # request data
        err_msg = e.swagger_result.get('error', {}).get('message', '')

        # would need to pass in exch, seems unnecessary
        # if 'balance' in err_msg.lower():
        #     # convert eg 6559678800 sats to 65.597 btc for easier reading
        #     n = re.search(r'(\d+)', err_msg)
        #     if n:
        #         n = n.group()
        #         err_msg = err_msg.replace(n, str(round(int(n) / self.div, 3)))

        #     m_bal = dict(
        #         avail_margin=self.avail_margin,
        #         total_balance_margin=self.total_balance_margin,
        #         total_balance_wallet=self.total_balance_wallet,
        #         unpl=self.unrealized_pnl)

        #     data['avail_balance'] = {k: round(v, 3) for k, v in m_bal.items()}

        super().__init__(
            request=request,
            code=e.status_code,
            api_message=err_msg,
            fail_msg=fail_msg)


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
        cur_qty='qty',
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

    order_keys = ExchOrder._m_conv['bitmex']
    api_host = 'https://www.bitmex.com'
    api_host_test = 'https://testnet.bitmex.com'
    api_spec = '/api/explorer/swagger.json'

    order_endpoints = dict(
        submit='Order.new',
        amend='Order.amend',
        cancel='Order.cancel',
        cancel_all='Order.cancelAll')

    def __init__(self, user: str, test: bool = False, refresh: bool = False, **kw):
        super().__init__(user=user, test=test, **kw)
        self.partialcandle = None

        jf.set_self(exclude=('refresh',))

        if refresh:
            self.refresh()

    @staticmethod
    def client_cls():
        return bitmex

    @staticmethod
    def client_api_auth():
        return BitmexAuth

    def req(
            self,
            request: str,
            fail_msg: str = None,
            **kw) -> Union[Any, Tuple[Any, int]]:
        """Build bitmex request
        - pass through super().check_request for request retries
        - bitmex returns specific parameter as a 400 with error message, not unique codes like bybit

        Parameters
        ----------
        request : str
            Endpoint.function, eg 'Funding.get'

        Returns
        -------
        Union[Any, Tuple[Any, int]]
            order data List[dict] etc, with optional ret_code
        """

        # request submitted as str, split it and call on client
        _request = self._make_request(request, **kw)

        try:
            return self.check_request(_request)
        except HTTPBadRequest as e:
            raise BitmexAPIException(e, _request, fail_msg) from None

    def _set_positions(self) -> List[dict]:
        positions = self.req('Position.get')

        # update position keys for easier access
        for p in positions:
            p |= {k: p[k] / self.div for k in self.wallet_keys.values() if k in p}

        return positions

    def set_leverage(self, symbol: str = None, lev: float = 7.0) -> None:
        symbol = symbol or self.default_symbol
        self.req('Position_updateLeverage', symbol=symbol, leverage=lev)

    def _get_instrument(self, **kw) -> dict:
        return self.req('Instrument.get', **kw)[0]

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

        orders = self.req(
            'Order.getOrders',
            filter=fltr,
            reverse=reverse,
            count=count,
            startTime=starttime)

        self._orders = self.add_custom_specs(orders)

    def next_funding(
            self,
            symbol: str = SYMBOL,
            with_hours: bool = False) -> Union[float, Tuple[float, int]]:
        """Get current funding rate from exchange"""
        result = self._get_instrument(symbol=symbol)

        rate = result['fundingRate']
        if not with_hours:
            return rate

        hrs = int((result['fundingTimestamp'] - f.inter_now().replace(tzinfo=tz.utc)).total_seconds() / 3600)

        return rate, hrs

    def _get_total_balance(self) -> dict:
        return self.req('User.getMargin', currency='XBt')

    def _route_order_request(self, action: str, order_specs: List[dict]):
        """Route bitmex order request

        Parameters
        ----------
        action : str
            submit | amend | cancel | cancel_all
        order_specs : List[dict]
            list of order dicts with keys converted to Bitmex keys
        """
        # could check/set exchange here
        endpoint = self.order_endpoints[action]

        if action == 'cancel':
            # cancel can still use bulk, just needs list of orderIds
            order_specs = [s['orderID'] for s in order_specs]
            order_specs = [{'orderID': json.dumps(order_specs)}]

        return_specs = []
        for spec in jf.as_list(order_specs):

            # cant amend with clOrdID, otherwise thinks trying to change it
            if action == 'amend':
                spec.pop('clOrdID')

            ret_spec = self.req(endpoint, **spec)
            jf.safe_append(return_specs, ret_spec)

        return return_specs

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

        lst = []

        for symbol, df in df.groupby('symbol'):

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

    def paged_req(self, request: str, page_max: int, pages: int = 100, **kw) -> List[dict]:
        """Batch calls into pages for eg candles/funding history
        - useful when updating full history
        - requests will sleep 10s if ratelim hit

        Parameters
        ----------
        request : str
            eg 'Trade.getBucketed'
        page_max : int
            max results per page for request
        pages : int, optional
            max num pages to get, by default 100

        Returns
        -------
        List[dict]
            list of data
        """
        resultcount = float('inf')
        start = 0
        data = []

        while resultcount >= page_max and start // page_max <= pages:
            _data = self.req(
                request,
                count=page_max,
                start=start,
                **kw)

            resultcount = len(_data)
            data.extend(_data)
            start += page_max

        return data

    def get_candles(
            self,
            symbol: str = SYMBOL,
            starttime: dt = None,
            fltr: str = '',
            retain_partial: bool = False,
            include_partial: bool = True,
            interval: int = 15,
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

        cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

        if isinstance(symbol, list) and fltr == '':
            fltr = json.dumps(dict(symbol=symbol))  # filter symbols needed
            symbol = None

        data = self.paged_req(
            'Trade.getBucketed',
            page_max=1000,
            binSize=binsize,
            symbol=symbol,
            startTime=starttime,
            filter=fltr,
            reverse=reverse,
            partial=include_partial,
            columns=json.dumps(cols))

        # fix bitmex high/low being less/greater than the open
        df = pd.DataFrame(data) \
            .assign(timestamp=lambda x: x.timestamp.dt.tz_localize(None) + offset * -1) \
            .pipe(self.resample, include_partial=include_partial, do=interval == 15) \
            .assign(
                high=lambda x: np.where(x.open > x.high, x.open, x.high),
                low=lambda x: np.where(x.open < x.low, x.open, x.low)) \
            .assign(interval=interval)[['interval'] + cols]

        if include_partial:
            self.partialcandle = df.tail(1).copy().reset_index(drop=True)  # save last as df

            if not retain_partial:
                df.drop(df.index[-1], inplace=True)

        return df

    def get_funding_history(self, symbol: str, starttime: dt) -> pd.DataFrame:
        """Get funding history for symbol
        - max count = 500

        Parameters
        ----------
        symbol : str

        Returns
        -------
        pd.DataFrame
            df with [symbol, timestamp]: [funding_rate]
        """
        cols = ['timestamp', 'symbol', 'fundingRate']
        data = self.paged_req(
            'Funding.get',
            page_max=500,
            filter=json.dumps(dict(symbol=jf.as_list(symbol))),
            startTime=starttime,
            columns=json.dumps(cols))

        if not data:
            log.warning(f'No funding data returned from Bitmex for symbols: {symbol}')
            return

        return pd.DataFrame(data)[cols] \
            .assign(timestamp=lambda x: x.timestamp.dt.tz_localize(None)) \
            .pipe(f.lower_cols) \
            .set_index(['symbol', 'timestamp'])

    def check_clock_skew(self) -> None:
        """Check local time vs exchange server for clock skew
        - locally seems to be ~1-4s average
        """
        pos = self.get_position('XBTUSD', refresh=True)
        t_server = pos['currentTimestamp'].timestamp()
        t_local = dt.utcnow().replace(tzinfo=tz.utc).timestamp()
        diff = t_server - t_local
        msg = f't_server: {t_server:.2f}\nt_local: {t_local:.2f}\ndiff: {diff:.2f}s'
        return msg
