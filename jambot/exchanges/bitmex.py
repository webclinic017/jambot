import itertools
import json
import time
import warnings
from collections import defaultdict as dd
from datetime import datetime as dt
from datetime import timedelta as delta
from datetime import timezone as tz
from typing import *

import pandas as pd
from bitmex import bitmex
from BitMEXAPIKeyAuthenticator import APIKeyAuthenticator

from jambot import SYMBOL, display
from jambot import functions as f
from jambot import getlog
from jambot.config import AZURE_WEB
from jambot.exchanges.exchange import SwaggerExchange
from jambot.tradesys import orders as ords
from jambot.tradesys.orders import ExchOrder, Order

log = getlog(__name__)
warnings.filterwarnings('ignore', category=Warning, message='.*format is not registered')


class Bitmex(SwaggerExchange):
    div = 100000000
    wallet_keys = dict(
        avail_margin='excessMargin',
        total_balance_margin='marginBalance',
        total_balance_wallet='walletBalance',
        unrealized_pnl='unrealisedPnl',
        prev_pnl='prevRealisedPnl')

    order_keys = dict(
        qty='orderQty',
        order_link_id='clOrdID'
    )

    api_host = 'https://www.bitmex.com'
    api_host_test = 'https://testnet.bitmex.com'
    api_spec = '/api/explorer/swagger.json'

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

    def _get_positions(self) -> List[dict]:
        return self.client.Position.Position_get().result()[0]

    def get_instrument(self, symbol: str = SYMBOL) -> dict:
        """Get symbol stats dict
        - add precision for rounding
        - Useful for getting precision or last price
        - no rest api delay

        Parameters
        ----------
        symbol : str, optional

        Returns
        -------
        dict
        """
        m = self.client.Instrument.Instrument_get(symbol=symbol).response().result[0]

        return m | dict(precision=len(str(m['tickSize']).split('.')[-1]))

    def last_price(self, symbol: str = SYMBOL) -> float:
        """Get last price for symbol (used for testing)
        - NOTE not currently used

        Parameters
        ----------
        symbol : str
            eg XBTUSD

        Returns
        -------
        float
            last price
        """
        return self.get_instrument(symbol=symbol)['lastPrice']

    def current_qty(self, symbol: str = None) -> Union[int, dict]:
        """Open contracts for each position

        Parameters
        ----------
        symbol : str, optional
            symbol to get current qty, by default None

        Returns
        -------
        Union[int, dict]
            single qty (if symbol given) or dict of all {symbol: currentQty}
        """
        if not symbol is None:
            return self.get_position(symbol).get('currentQty', 0)
        else:
            # all position qty
            return {k: v['currentQty'] for k, v in self._positions.items()}

    def df_orders(self, symbol: str = SYMBOL, new_only: bool = True, refresh: bool = False) -> pd.DataFrame:
        """Used to display orders in google sheet

        Parameters
        ----------
        symbol : str, optional
            default SYMBOL
        new_only : bool, optional
            default True
        refresh : bool, optional
            default False

        Returns
        -------
        pd.DataFrame
            df of orders
        """
        orders = self.get_orders(symbol=symbol, new_only=new_only, refresh=refresh, as_exch_order=True)
        cols = ['order_type', 'name', 'qty', 'price', 'exec_inst', 'symbol']

        if not orders:
            df = pd.DataFrame(columns=cols, index=range(1))
        else:
            data = [{k: o.raw_spec(k) for k in cols} for o in orders]
            df = pd.DataFrame.from_dict(data) \

        return df \
            .reindex(columns=cols) \
            .sort_values(
                by=['symbol', 'order_type', 'name'],
                ascending=[False, True, True])

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

    def get_orders(
            self,
            symbol: str = None,
            new_only: bool = False,
            bot_only: bool = False,
            manual_only: bool = False,
            as_exch_order: bool = True,
            as_dict: bool = False,
            refresh: bool = False) -> Union[List[dict], List[ExchOrder], Dict[str, ExchOrder]]:
        """Get orders which match criterion

        Parameters
        ----------
        symbol : str, optional
            symbol eg 'xbtusd'
        new_only : bool, optional
            unfilled orders, default False
        bot_only : bool, optional
            orders created by bot, default False
        manual_only : bool, optional
            orders not created by bot, by default False
        as_exch_order : bool, optional
            return BitmexOrders instead of raw list of dicts, default True
        as_dict : bool, optional
            return Dict[str, BitmexOrders] instead of list (for matching)
        refresh : bool, optional
            run self.set_orders() first to ensure orders current with bitmex

        Returns
        -------
        Union[List[dict], List[ExchOrder] Dict[str, ExchOrder]]
            list of RAW order dicts, list of bitmex orders, or dict of str: ExchOrder
        """
        if refresh or self._orders is None:
            self.set_orders()

        var = {k: v for k, v in vars().items() if not v in ('as_exch_order', 'refresh', 'as_dict')}

        # define filters
        conds = dict(
            symbol=lambda x: x['symbol'].lower() == symbol.lower(),
            new_only=lambda x: x['ordStatus'] in ('New', 'PartiallyFilled'),
            bot_only=lambda x: not x['manual'],
            manual_only=lambda x: x['manual'])

        # filter conditions based on true args in vars()
        conds = {k: v for k, v in conds.items() if not var.get(k) in (None, False)}

        orders = [o for o in self._orders if all(cond(o) for cond in conds.values())]

        if as_exch_order:
            orders = self.exch_order_from_raw(order_specs=orders, process=False)

        if as_dict:
            orders = ords.list_to_dict(orders, key_base=False)

        return orders

    def exch_order_from_raw(self, order_specs: list, process: bool = True) -> List[ExchOrder]:
        """Create bitmex order objs from raw specs
        - top level wrapper to both add raw specs and convert to ExchOrder

        Parameters
        ----------
        order_specs : list
            list of order specs from bitmex

        Returns
        -------
        List[ExchOrder]
            list of bitmex orders
        """
        if process:
            order_specs = self.add_custom_specs(order_specs)

        # TODO temp solution should handle depending on result
        # order_specs is None
        if order_specs is None:
            log.warning('order_specs is None.')

        for item in order_specs:
            if not isinstance(item, dict):
                raise AttributeError(f'Invalid order specs returned from bitmex. {type(item)}: {item}')

        return ords.make_exch_orders(order_specs)

    def set_orders(
            self,
            fltr: dict = None,
            count: int = 100,
            starttime: dt = None,
            reverse: bool = True) -> None:
        """Save last count orders from exchange"""

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
        """Get current funding rate from bitmex"""
        result = self.client.Instrument.Instrument_get(symbol=symbol).response().result[0]

        rate = result['fundingRate']
        hrs = int((result['fundingTimestamp'] - f.timenow().replace(tzinfo=tz.utc)).total_seconds() / 3600)

        return rate, hrs

    def _get_total_balance(self) -> dict:
        return self.check_request(self.client.User.User_getMargin(currency='XBt'))

    def _order_request(self, action: str, order_specs: list) -> Union[List[ExchOrder], None]:
        """Send order submit/amend/cancel request

        Parameters
        ----------
        action : str
            submit | amend | cancel
        order_specs : list
            list of orders to process

        Returns
        -------
        Union[List[ExchOrder], None]
            list of order results or None if request failed
        """
        if not order_specs and not action == 'cancel_all':
            return

        func, param = dict(
            submit=('Order_newBulk', 'orders'),
            amend=('Order_amendBulk', 'orders'),
            cancel=('Order_cancel', 'orderID'),
            cancel_all=('Order_cancelAll', '')).get(action)

        func = getattr(self.client.Order, func)

        if not action == 'cancel_all':
            params = {param: json.dumps(order_specs)}
        else:
            # cancel_all just uses dict(symbol='XBTUSD')
            params = order_specs

        result = self.check_request(func(**params))

        # result can be None if bad orders
        if result is None:
            return

        resp_orders = self.exch_order_from_raw(order_specs=result)

        # check if submit/amend orders incorrectly cancelled
        if not 'cancel' in action:
            failed_orders = [o for o in resp_orders if o.is_cancelled]

            if failed_orders:
                msg = 'ERROR: Order(s) CANCELLED!'
                for o in failed_orders:
                    err_text = o.raw_spec('text')
                    m = o.order_spec

                    # failed submitted at offside price, add last_price for comparison
                    if 'ParticipateDoNotInitiate' in err_text:
                        m['last_price'] = self.last_price(symbol=o.symbol)

                    msg += f'\n\n{err_text}\n{f.pretty_dict(m, prnt=False, bold_keys=True)}'

                f.discord(msg, channel='err')

        return resp_orders

    def amend_orders(self, orders: Union[List[ExchOrder], ExchOrder]) -> List[ExchOrder]:
        """Amend order price and/or qty

        Parameters
        ----------
        orders : Union[List[ExchOrder], ExchOrder]
            Orders to amend price/qty

        Returns
        -------
        List[ExchOrder]
        """
        order_specs = [o.order_spec_amend for o in f.as_list(orders)]
        return self._order_request(action='amend', order_specs=order_specs)

    def submit_simple_order(
            self,
            qty: int,
            price: float = None,
            ordtype: str = 'Limit',
            symbol: str = SYMBOL):
        """Qucik submit single order from args instead of order obj"""

        if price is None:
            ordtype = 'Market'

        order = ExchOrder(
            price=price,
            qty=qty,
            order_type=ordtype,
            symbol=symbol)

        print('Sending new_order:')
        display(order.order_spec())

        o = self.place_bulk(placeorders=order)[0]
        display(f.useful_keys(o))

    def submit_orders(self, orders: Union[List[ExchOrder], ExchOrder, List[dict]]) -> List[ExchOrder]:
        """Submit single or multiple orders

        Parameters
        ----------
        orders : List[ExchOrder] | ExchOrder | List[dict]
            single or list of ExchOrder objects, or list/single dict of order specs

        Returns
        -------
        list
            list of orders placed
        """
        _orders = []
        for o in f.as_list(orders):
            # convert dict specs to ExchOrder
            # NOTE ExchOrder IS a subclass of dict!!
            if not isinstance(o, ExchOrder):

                # isinstance(o, Order) doesn't want to work
                if hasattr(o, 'as_exch_order'):
                    o = o.as_exch_order()
                else:
                    # dict
                    o = ords.make_orders(order_specs=o, as_exch_order=True)[0]

            _orders.append(o)

        # remove orders with zero qty
        orders = [o for o in _orders if not o.qty == 0]
        if not orders:
            return

        # split orders into groups, market orders must be submitted as single
        grouped = {k: list(g) for k, g in itertools.groupby(orders, lambda x: x.is_market)}

        batches = []

        # add each market order as it's own group - send market orders first
        for order in grouped.get(True, []):
            batches.append([order])

        # limit orders
        batches.append(grouped.get(False, []))

        result = []
        for order_batch in batches:
            result_orders = self._order_request(action='submit', order_specs=order_batch)
            result.extend(result_orders or [])

        return result

    def close_position(self, symbol: str = SYMBOL) -> None:
        """Market close current open position

        Parameters
        ----------
        symbol : str, optional
            symbol to close position, by default 'XBTUSD'
        """
        try:
            self.check_request(
                self.client.Order.Order_new(symbol=symbol, execInst='Close'))
        except:
            f.send_error(msg='ERROR: Could not close position!')

    def cancel_manual(self) -> List[ExchOrder]:
        raise NotImplementedError('this doesn\'t work yet.')
        orders = self.get_orders(refresh=True, manual_only=True)
        self.cancel_orders(orders=orders)

    def cancel_orders(self, orders: Union[List[ExchOrder], ExchOrder]) -> List[ExchOrder]:
        """Cancel one or multiple orders by order_id

        Parameters
        ----------
        orders : List[ExchOrder] | ExchOrder
            list of orders to cancel

        Returns
        -------
        List[ExchOrder]
            list of cancelled orders
        """
        order_specs = [o.order_id for o in f.as_list(orders)]

        if not order_specs:
            log.warning('No orders to cancel.')
            return

        return self._order_request(action='cancel', order_specs=order_specs)

    def cancel_all_orders(self, symbol: str = None) -> List[ExchOrder]:
        """Cancel all open orders

        Parameters
        ----------
        symbol : str, default cancel all symbols
            specific symbol to filter cancel

        Returns
        -------
        List[ExchOrder]
            list of cancelled orders
        """
        return self._order_request(action='cancel_all', order_specs=dict(symbol=symbol))

    def get_partial(self, symbol):
        timediff = 0

        if not self.partialcandle is None:
            timediff = (self.partialcandle.timestamp[0] - f.timenow()).seconds

        if (timediff > 7200
            or self.partialcandle is None
                or not self.partialcandle.symbol[0] == symbol):
            self.set_partial(symbol=symbol)

        return self.partialcandle

    def set_partial(self, symbol):
        # call only partial candle from bitmex, save to self.partialcandle
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
        """Wait till last period candle is available from bitmex
            - rest api latency is ~15s
            - Just for testing, not used for anything

        Parameters
        ----------
        interval : int, optional
            default 1
        """

        # set current lower time bin to check for
        dnow = f.timenow(interval) - f.get_offset(interval)

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
        lst = []
        cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

        while resultcount >= 1000 and start // 1000 <= pages:
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
            start += 1000

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

    def reconcile_orders(self, symbol: str, expected_orders: List[Order], discord_user: str = None) -> None:
        """Compare expected and actual (current) orders, adjust as required

        Parameters
        ----------
        expected_orders : List[Order]
            orders from current timestamp in strat backtest
        actual_orders : List[Order]
            orders active on exchange
        """
        actual_orders = self.get_orders(symbol=symbol, new_only=True, bot_only=True, as_exch_order=True, refresh=True)
        all_orders = self.validate_orders(expected_orders, actual_orders, show=True)

        # perform action reqd for orders except valid/manual
        for action, orders in all_orders.items():
            if orders and not action in ('valid', 'manual'):
                getattr(self, f'{action}_orders')(orders)

        # temp send order submit details to discord
        user = self.user if discord_user is None else discord_user
        # m = dict(user=user)
        m = {}
        m_ords = {k: [o.short_stats for o in orders]
                  for k, orders in all_orders.items() if orders and not k == 'manual'}

        if m_ords:
            m['current_qty'] = f'{self.current_qty(symbol=symbol):+,}'
            m |= m_ords
            msg = f.pretty_dict(m, prnt=False, bold_keys=True)
            f.discord(msg=f'{user}\n{msg}', channel='orders')

        s = ', '.join([f'{action}={len(orders)}' for action, orders in all_orders.items()])
        log.info(f'{self.user} - Reconciled orders: {s}')

    def validate_orders(
            self,
            expected_orders: List[Order],
            actual_orders: List[ExchOrder],
            show: bool = False) -> Dict[str, List[Order]]:
        """Compare expected and actual (current) orders, return reqd actions

        Parameters
        ----------
        expected_orders : List[Order]
            orders from current timestamp in strat backtest
        actual_orders : List[ExchOrder]
            orders active on exchange
        show : bool
            display df of orders with reqd action (for testing)

        Returns
        -------
        Dict[str, List[Order]]
            - amend: (orders matched but price/qty different)
            - valid: (orders matched and have correct price/qty - no action reqd)
            - submit: (expected_order missing)
            - cancel: (actual_order not found in expected_orders)
            - manual: (manually placed order, don't touch)
        """

        # convert to dicts for easier matching
        expected_orders = ords.list_to_dict(expected_orders)
        actual_orders = ords.list_to_dict(actual_orders)
        all_orders = dd(list, {k: [] for k in ('valid', 'cancel', 'amend', 'submit', 'manual')})

        for k, o in actual_orders.items():
            if k in expected_orders:
                o_expc = expected_orders[k]

                # compare price/qty
                if not (o.price == o_expc.price and o.qty == o_expc.qty):
                    o.amend_from_order(o_expc)
                    all_orders['amend'].append(o)
                else:
                    all_orders['valid'].append(o)
            else:
                if o.is_manual:
                    all_orders['manual'].append(o)
                else:
                    all_orders['cancel'].append(o)

        all_orders['submit'] = [o for k, o in expected_orders.items() if not k in actual_orders]

        if show and not AZURE_WEB:
            dfs = []
            for action, orders in all_orders.items():
                if orders:
                    data = [o.to_dict() for o in orders]
                    df = pd.DataFrame.from_dict(data) \
                        .assign(action=action)

                    dfs.append(df)

            if dfs:
                display(pd.concat(dfs))

        return all_orders
