import itertools
import json
import re
import warnings
from collections import defaultdict as dd

from bitmex import bitmex
from swagger_spec_validator.common import SwaggerValidationWarning

from jambot import AZURE_WEB
from jambot import functions as f
from jambot.exchanges.exchange import Exchange
from jambot.tradesys import orders as ords
from jambot.tradesys.__init__ import *
from jambot.tradesys.orders import BitmexOrder, Order
from jambot.utils.secrets import SecretsManager

log = getlog(__name__)
warnings.filterwarnings('ignore', category=Warning, message='.*format is not registered')


class Bitmex(Exchange):
    div = 100000000

    def __init__(self, user: str, test: bool = False, refresh: bool = False, **kw):
        super().__init__(user=user, test=test, **kw)
        # self.name = ''
        # self.nameshort = ''
        self.balance_set = False
        self.avail_margin = 0
        self.total_balance_margin = 0
        self.total_balance_wallet = 0
        self.reserved_balance = 0
        self.unrealized_pnl = 0
        self.prev_pnl = 0
        self._orders = None
        self._positions = None
        self.partialcandle = None

        f.set_self(vars(), exclude=('refresh',))

        if refresh:
            self.refresh()

    @classmethod
    def default(cls, test: bool = True, refresh: bool = True, **kw) -> 'Bitmex':
        """Create Bitmex obj with default name"""
        user = 'jayme' if not test else 'testnet'
        return cls(user=user, test=test, refresh=refresh, **kw)

    def load_creds(self, user: str):
        """Load creds from csv"""
        df = SecretsManager('bitmex.csv').load \
            .set_index('user')

        if not user in df.index:
            raise RuntimeError(f'User "{user}" not in saved credentials.')

        return df.loc[user].to_dict()

    def init_client(self, test: bool = False):
        warnings.simplefilter('ignore', SwaggerValidationWarning)
        return bitmex(test=test, api_key=self.key, api_secret=self.secret)

    def refresh(self):
        """Set margin balance, current position info, all orders"""
        self.set_total_balance()
        self.set_positions()
        self.set_orders()

    @property
    def orders(self):
        return self._orders

    @property
    def balance(self) -> float:
        """Get current exchange balance in Xbt, minus user-defined "reserved" balance
        - "Wallet Balance" on bitmex, NOT "Available Balance"

        Returns
        -------
        float
        """
        return self.total_balance_wallet - self.reserved_balance

    def check_request(self, request, retries: int = 0):
        """
        Perform http request and retry with backoff if failed

        - type(request) = bravado.http_future.HttpFuture
        - type(response) = bravado.response.BravadoResponse
        - response = request.response(fallback_result=[400], exceptions_to_catch=bravado.exception.HTTPBadRequest)
        """
        # bravado.exception.HTTPBadRequest

        try:
            # 400 HttpBadRequest raised at response()
            response = request.response(fallback_result='')
            status = response.metadata.status_code
            backoff = 0.5

            if status < 300:
                return response.result
            elif status == 503 and retries < 7:
                retries += 1
                sleeptime = backoff * (2 ** retries - 1)
                time.sleep(sleeptime)
                return self.check_request(request=request, retries=retries)
            else:
                f.send_error('{}: {}\n{}'.format(status, response.result, request.future.request.data))

        except Exception as e:
            data = request.future.request.data
            err_msg = e.swagger_result.get('error', {}).get('message', '')

            if 'balance' in err_msg.lower():
                # convert eg 6559678800 sats to 65.597 btc for easier reading
                n = re.search(r'(\d+)', err_msg)
                if n:
                    n = n.group()
                    err_msg = err_msg.replace(n, str(round(int(n) / self.div, 3)))

                m_bal = dict(
                    avail_margin=self.avail_margin,
                    total_balance_margin=self.total_balance_margin,
                    total_balance_wallet=self.total_balance_wallet,
                    unpl=self.unrealized_pnl)

                data['avail_balance'] = {k: round(v, 3) for k, v in m_bal.items()}

            # request data is dict of {key: str(list(dict))}
            # need to deserialize inner items first
            if isinstance(data, dict):
                m = {k: json.loads(v) if isinstance(v, str) else v for k, v in data.items()}
                data = f.pretty_dict(m=m, prnt=False, bold_keys=True)

            if AZURE_WEB:
                f.send_error(f'{e.status_code} {e.__class__.__name__}: {err_msg}\n{data}')
            else:
                raise e

    def get_position(self, symbol: str) -> dict:
        """Get position for specific symbol"""
        return self._positions.get(symbol.lower())

    def set_positions(self) -> None:
        """Set position for all symbols"""
        res = self.client.Position.Position_get().result()[0]

        # store as dict with symbol as keys
        self._positions = {}

        for pos in res:
            symbol = pos['symbol']
            self._positions[symbol.lower()] = pos

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
            return self.get_position(symbol)['currentQty']
        else:
            # all position qty
            return {k: v['currentQty'] for k, v in self._positions.items()}

    def df_orders(self, symbol: str = SYMBOL, new_only: bool = True, refresh: bool = False) -> pd.DataFrame:
        orders = self.get_orders(symbol=symbol, new_only=new_only, refresh=refresh, as_bitmex=True)
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

    def get_order_by_key(self, key):
        if self._orders is None:
            self.set_orders(new_only=True)

        # Manual orders don't have a key, not unique
        orders = list(filter(lambda x: 'key' in x.keys(), self._orders))
        orders = list(filter(lambda x: x['key'] == key, orders))

        if orders:
            return orders[0]  # assuming only one order will match given key
        else:
            return dd(type(None))

    def get_filled_orders(self, symbol: str = SYMBOL, starttime: dt = None) -> List[BitmexOrder]:
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
        List[BitmexOrder]
            list of recently filled orders
        """
        if starttime is None:
            starttime = dt.utcnow() + delta(days=-7)

        fltr = dict(ordStatus=['Filled', 'PartiallyFilled'])

        self.set_orders(fltr=fltr, starttime=starttime, reverse=False)
        return self.bitmex_order_from_raw(order_specs=self._orders, process=False)

    def get_orders(
            self,
            symbol: str = None,
            new_only: bool = False,
            bot_only: bool = False,
            manual_only: bool = False,
            as_bitmex: bool = True,
            as_dict: bool = False,
            refresh: bool = False) -> Union[List[dict], List[BitmexOrder], Dict[str, BitmexOrder]]:
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
        as_bitmex : bool, optional
            return BitmexOrders instead of raw list of dicts, default True
        as_dict : bool, optional
            return Dict[str, BitmexOrders] instead of list (for matching)
        refresh : bool, optional
            run self.set_orders() first to ensure orders current with bitmex

        Returns
        -------
        Union[List[dict], List[BitmexOrder] Dict[str, BitmexOrder]]
            list of RAW order dicts, list of bitmex orders, or dict of str: BitmexOrder
        """
        if refresh or self._orders is None:
            self.set_orders()

        var = {k: v for k, v in vars().items() if not v in ('as_bitmex', 'refresh', 'as_dict')}

        # define filters
        conds = dict(
            symbol=lambda x: x['symbol'].lower() == symbol.lower(),
            new_only=lambda x: x['ordStatus'] in ('New', 'PartiallyFilled'),
            bot_only=lambda x: not x['manual'],
            manual_only=lambda x: x['manual'])

        # filter conditions based on true args in vars()
        conds = {k: v for k, v in conds.items() if not var.get(k) in (None, False)}

        orders = [o for o in self._orders if all(cond(o) for cond in conds.values())]

        if as_bitmex:
            orders = self.bitmex_order_from_raw(order_specs=orders, process=False)

        if as_dict:
            orders = ords.list_to_dict(orders, key_base=False)

        return orders

    def bitmex_order_from_raw(self, order_specs: list, process: bool = True) -> List[BitmexOrder]:
        """Create bitmex order objs from raw specs
        - top level wrapper to both add raw specs and convert to BitmexOrder

        Parameters
        ----------
        order_specs : list
            list of order specs from bitmex

        Returns
        -------
        List[BitmexOrder]
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

        return ords.make_bitmex_orders(order_specs)

    def add_custom_specs(self, order_specs: List[dict]) -> List[dict]:
        """Preprocess orders from exchange to add custom markers

        Parameters
        ----------
        order_specs : List[dict]
            list of order specs from bitmex

        Returns
        -------
        List[dict]
            list of order specs with custom keys added

        Raises
        ------
        RuntimeError
            if order has already been processed
        """
        if order_specs is None:
            log.warning('No orders to add custom specs to.')
            return

        for o in f.as_list(order_specs):
            if 'processed' in o.keys():
                raise RuntimeError('Order spec has already been processed!')

            o['processed'] = True
            o['sideStr'] = o['side']
            o['side'] = 1 if o['side'] == 'Buy' else -1

            # some orders can have none qty if "close position" market order cancelled by bitmex
            if not o['orderQty'] is None:
                o['orderQty'] = int(o['side'] * o['orderQty'])

            # add key to the order, excluding manual orders
            if not o['clOrdID'] == '':
                # o['name'] = '-'.join(o['clOrdID'].split('-')[1:-1])
                o['name'] = o['clOrdID'].split('-')[1]

                if not 'manual' in o['clOrdID']:
                    # replace 10 digit timestamp from key if exists
                    o['key'] = re.sub(r'-\d{10}', '', o['clOrdID'])
                    # print(o['key'])
                    o['manual'] = False
                else:
                    o['manual'] = True
            else:
                o['name'] = '(manual)'
                o['manual'] = True

        return order_specs

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

    def set_total_balance(self) -> None:
        """Set margin/wallet values"""
        div = self.div
        res = self.check_request(self.client.User.User_getMargin(currency='XBt'))
        # total available/unused > only used in postOrder "Available Balance"
        self.avail_margin = res['excessMargin'] / div
        self.total_balance_margin = res['marginBalance'] / div  # unrealized + realized > wallet total qty avail
        self.total_balance_wallet = res['walletBalance'] / div  # realized
        self.unrealized_pnl = res['unrealisedPnl'] / div
        self.prev_pnl = res['prevRealisedPnl'] / div
        self.balance_set = True
        self.res = res

    def _order_request(self, action: str, order_specs: list) -> Union[List[BitmexOrder], None]:
        """Send order submit/amend/cancel request

        Parameters
        ----------
        action : str
            submit | amend | cancel
        order_specs : list
            list of orders to process

        Returns
        -------
        Union[List[BitmexOrder], None]
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

        resp_orders = self.bitmex_order_from_raw(order_specs=result)

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

    def amend_orders(self, orders: Union[List[BitmexOrder], BitmexOrder]) -> List[BitmexOrder]:
        """Amend order price and/or qty

        Parameters
        ----------
        orders : Union[List[BitmexOrder], BitmexOrder]
            Orders to amend price/qty

        Returns
        -------
        List[BitmexOrder]
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

        order = BitmexOrder(
            price=price,
            qty=qty,
            order_type=ordtype,
            symbol=symbol)

        print('Sending new_order:')
        display(order.order_spec())

        o = self.place_bulk(placeorders=order)[0]
        display(f.useful_keys(o))

    def submit_orders(self, orders: Union[List[BitmexOrder], BitmexOrder, List[dict]]) -> List[BitmexOrder]:
        """Submit single or multiple orders

        Parameters
        ----------
        orders : List[BitmexOrder] | BitmexOrder | List[dict]
            single or list of BitmexOrder objects, or list/single dict of order specs

        Returns
        -------
        list
            list of orders placed
        """
        _orders = []
        for o in f.as_list(orders):
            # convert dict specs to BitmexOrder
            # NOTE BitmexOrder IS a subclass of dict!!
            if not isinstance(o, BitmexOrder):

                # isinstance(o, Order) doesn't want to work
                if hasattr(o, 'as_bitmex'):
                    o = o.as_bitmex()
                else:
                    # dict
                    o = ords.make_orders(order_specs=o, as_bitmex=True)[0]

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

    def cancel_manual(self) -> List[BitmexOrder]:
        raise NotImplementedError('this doesn\'t work yet.')
        orders = self.get_orders(refresh=True, manual_only=True)
        self.cancel_orders(orders=orders)

    def cancel_orders(self, orders: Union[List[BitmexOrder], BitmexOrder]) -> List[BitmexOrder]:
        """Cancel one or multiple orders by order_id

        Parameters
        ----------
        orders : List[BitmexOrder] | BitmexOrder
            list of orders to cancel

        Returns
        -------
        List[BitmexOrder]
            list of cancelled orders
        """
        order_specs = [o.order_id for o in f.as_list(orders)]

        if not order_specs:
            log.warning('No orders to cancel.')
            return

        return self._order_request(action='cancel', order_specs=order_specs)

    def cancel_all_orders(self, symbol: str = None) -> List[BitmexOrder]:
        """Cancel all open orders

        Parameters
        ----------
        symbol : str, default cancel all symbols
            specific symbol to filter cancel

        Returns
        -------
        List[BitmexOrder]
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
            symbol: str = '',
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

    def reconcile_orders(self, symbol: str, expected_orders: List[Order]) -> None:
        """Compare expected and actual (current) orders, adjust as required

        Parameters
        ----------
        expected_orders : List[Order]
            orders from current timestamp in strat backtest
        actual_orders : List[Order]
            orders active on exchange
        """
        actual_orders = self.get_orders(symbol=symbol, new_only=True, bot_only=True, as_bitmex=True, refresh=True)
        all_orders = self.validate_orders(expected_orders, actual_orders, show=True)

        # perform action reqd for orders except valid/manual
        for action, orders in all_orders.items():
            if orders and not action in ('valid', 'manual'):
                getattr(self, f'{action}_orders')(orders)

        # temp send order submit details to discord
        m = dict(user=self.user)
        m_ords = {k: [o.short_stats for o in orders]
                  for k, orders in all_orders.items() if orders and not k == 'manual'}

        if m_ords:
            m['current_qty'] = f'{self.current_qty(symbol=symbol):+,}'
            m |= m_ords
            msg = f.pretty_dict(m, prnt=False, bold_keys=True)
            f.discord(msg=msg, channel='orders')

        s = ', '.join([f'{action}={len(orders)}' for action, orders in all_orders.items()])
        log.info(f'{self.user} - Reconciled orders: {s}')

    def validate_orders(
            self,
            expected_orders: List[Order],
            actual_orders: List[BitmexOrder],
            show: bool = False) -> Dict[str, List[Order]]:
        """Compare expected and actual (current) orders, return reqd actions

        Parameters
        ----------
        expected_orders : List[Order]
            orders from current timestamp in strat backtest
        actual_orders : List[BitmexOrder]
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
