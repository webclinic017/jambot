import time
from datetime import datetime as dt
from datetime import timedelta as delta
from datetime import timezone as tz
from typing import *

import pandas as pd

from jambot import comm as cm
from jambot import functions as f
from jambot import getlog
from jambot.exchanges.bybit_compat import APIKeyAuthenticator, bybit
from jambot.exchanges.exchange import SwaggerAPIException, SwaggerExchange
from jambot.tradesys.orders import ExchOrder
from jgutils import functions as jf
from jgutils import pandas_utils as pu

if TYPE_CHECKING:
    from bravado.http_future import HttpFuture

SYMBOl = 'BTCUSD'

log = getlog(__name__)

# TODO make sure balance_margin set for linear usdt contracts properly

"""
ByBit API issues
----------------
- Position
    - Positions_myPosition doesn't return last_price (can't calculate position PNL)
    - inconsistent value types returned (string/float) eg '0.99768569' or 5.259e-05

- Orders
    - conditional_cancel_all still uses bitmex clOrdID (LOL IDIOTS)
    - conditional_submit doesn't return order_status
    - conditional_submit returns order type, but not stop_order_type
    - inconsistent price/qty return types (sometimes string, sometimes int)
    - amend/cancel only return order_id (need to query again)
        - but cancel_all returns all fields (except for Linear)
    - stops have completely different and inconsistent endpoint than market/limit
    - update/created_at times return inconsistent length microsec decimals (or none)
        - Linear returns updated_time, created_time
    - stop orders have "stop_order_id" instead of just "order_id"
    - Conditional_query doesnt have any fields to be able to determine its conditional
    - no way to filter orders by filled_time
"""


class BybitAuth(APIKeyAuthenticator):
    """
    Wrap BybitAPIKeyAuthenticator to add longer expiry timeout
    """

    def apply(self, r):
        r.headers['User-Agent'] = 'Official-SDKs'
        expires = str(int(round(time.time()) - 1)) + '000'
        r.params['timestamp'] = expires
        r.params['recv_window'] = 20_000
        r.params['api_key'] = self.api_key
        r.params['sign'] = self.generate_signature(r)
        return r


class BybitAPIException(SwaggerAPIException):
    def __init__(
            self,
            request: 'HttpFuture',
            result: dict,
            fail_msg: str = None) -> None:
        """Raise exception on bybit invalid api request

        Parameters
        ----------
        request : HttpFuture
            original request to get operation name
        result : dict
            bybit api dict with error response data
        fail_msg : str, optional
            custom additional err info message, by default None
        """
        super().__init__(
            request=request,
            code=result['ret_code'],
            api_message=result['ret_msg'],
            fail_msg=fail_msg)


class Bybit(SwaggerExchange):
    div = 1
    default_symbol = 'BTCUSD'
    conv_symbol = dict(XBTUSD=default_symbol)
    # TODO confirm these
    wallet_keys = dict(
        avail_margin='available_balance',
        total_balance_margin='wallet_balance',  # position_margin?
        total_balance_wallet='wallet_balance',
        unrealized_pnl='unrealised_pnl',
        prev_pnl='realised_pnl')

    other_keys = dict(
        last_price='last_price',
        cur_qty='qty',
        err_text='reject_reason')

    pos_keys = dict(
        qty='size',
        u_pnl='unrealised_pnl',
        r_pnl='realised_pnl',
        value='position_value')

    order_keys = ExchOrder._m_conv['bybit']
    api_host = 'https://api.bybit.com'
    api_host_test = 'https://api-testnet.bybit.com'
    api_spec = '/doc/swagger/v_0_2_12.txt'

    order_endpoints = dict(
        submit='Order.new',
        amend='Order.replace',
        cancel='Order.cancel',
        cancel_all='Order.cancelAll')

    def __init__(self, user: str, test: bool = False, refresh: bool = False, **kw):
        super().__init__(user=user, test=test, **kw)

        if refresh:
            self.refresh()

    @staticmethod
    def client_cls():
        return bybit

    @staticmethod
    def client_api_auth():
        return BybitAuth

    def _get_total_balance(self) -> dict:
        return self.req('Wallet.getBalance', coin='BTC')['BTC']

    def req(
            self,
            request: Union['HttpFuture', str],
            code: bool = False,
            fail_msg: str = None,
            **kw) -> Union[Any, Tuple[Any, int], Dict[str, Any], List[Dict[str, Any]]]:
        """Wrapper to handle bybit request response/error code
        - pass through super().check_request for request retries

        Example return spec
        -------------------
        {'ret_code': 0,
            'ret_msg': 'OK',
            'ext_code': '',
            'ext_info': '',
            'result': {'data': [{'user_id': 288389,
                'position_idx': 0,
                ...

        Parameters
        ----------
        request : HttpFuture
            http request to submit to bybit
        code : bool, optional
            return error code or not, by default False

        Returns
        -------
        Union[Any, Tuple[Any, int]]
            order data List[dict] etc, with optional ret_code
        """

        # request submitted as str, split it and call on client
        if isinstance(request, str):
            request = self._make_request(request, **kw)

        full_result = self.check_request(request)
        ret_code = full_result['ret_code']

        if not ret_code == 0:
            raise BybitAPIException(request, full_result, fail_msg)

        result = full_result['result']

        if isinstance(result, dict) and 'data' in result:
            result = result['data']

            # UGH Conditional returns data=[], LinearConditional returns data=None
            if result is None:
                result = []

        # default, just give the data
        return result if not code else (result, ret_code)

    def set_orders(self, symbol: str = SYMBOl, bybit_async: bool = False, bybit_stops: bool = False):
        """set raw order dicts from exch
        limits: max = 50, default = 20
        - https://bybit-exchange.github.io/docs/inverse/#t-getactive

        Parameters
        ----------
        symbol : str, optional
            default BTCUSD
        bybit_async : bool, optional
            order submission/creation is async so "all" orders are delayed, default False
        bybit_stops : bool, optional
            bybit keeps all stops under "conditional" endpoint
        """
        def _filter_keys(existing: List[dict], new: List[dict]) -> List[dict]:
            """don't add async orders to final if already exist in regular"""
            k = 'order_link_id'
            existing_keys = [o[k] for o in existing]
            return [o for o in new if not o[k] in existing_keys]

        prefix = 'Linear' if symbol.lower().endswith('usdt') else ''
        kw = dict(symbol=symbol, limit='30')  # TODO not sure if want this limit always
        orders = self.req(f'{prefix}Order.getOrders', **kw)

        if bybit_async:
            orders += _filter_keys(orders, self.req(f'{prefix}Order.query', **kw))

        # very sketch, bybit returns stop orders as "Market", with no price
        orders = [o for o in orders if not 'stop' in o['order_link_id']]

        stop_orders = []
        if bybit_stops:
            stop_orders = self.req(f'{prefix}Conditional.getOrders', **kw)

            if bybit_async:
                stop_orders += _filter_keys(stop_orders, self.req(f'{prefix}Conditional.query', **kw))

        self._orders = self.add_custom_specs(self.proc_raw_spec(orders + stop_orders))

    def get_active_instruments(self, **kw) -> List[Dict[str, Any]]:
        """Get all open symbol data

        Examples
        --------
        [{'name': 'SOLUSDT',
        'alias': 'SOLUSDT',
        'status': 'Trading',
        'base_currency': 'SOL',
        'quote_currency': 'USDT',
        'price_scale': 3,
        'taker_fee': '0.00075',
        'maker_fee': '-0.00025',
        'leverage_filter':
            {'min_leverage': 1,
            'max_leverage': 50,
            'leverage_step': '0.01'},
        'price_filter':
            {'min_price': '0.005',
            'max_price': '9999.99',
            'tick_size': '0.005'},
        'lot_size_filter':
            {'max_trading_qty': 3000,
            'min_trading_qty': 0.1,
            'qty_step': 0.1}}]

        Returns
        -------
        List[Dict[str, Any]]
            [description]
        """
        data = self.req('Symbol.get', **kw)  # type: List[Dict[str, Any]]
        data_out = []

        for m_in in data:
            if m_in['status'] == 'Trading':
                m = {}
                m['symbol'] = m_in['name']
                m['base_currency'] = m_in['base_currency']
                m['quote_currency'] = m_in['quote_currency']
                m['is_inverse'] = not m_in['name'].lower().endswith('usdt')
                m['tick_size'] = float(m_in['price_filter']['tick_size'])
                m['lot_size'] = float(m_in['lot_size_filter']['min_trading_qty'])

                data_out.append(m)

        return data_out

    def _set_positions(self) -> List[dict]:
        """Get position info, eg current qty

        Returns
        -------
        List[dict]
            dicts of all positions
        """
        positions = []

        # UGH linear positions are different endpoint
        for prefix in ('Linear', ''):
            for m in self.req(f'{prefix}Positions.myPosition'):
                p = {k: f.str_to_num(v) for k, v in m['data'].items()}
                p['side_str'] = p['side']
                p['side'] = {'Buy': 1, 'Sell': -1}.get(p['side'])
                p['size'] = p['size'] * (p['side'] or 1)
                p['sym_short'] = p['symbol'][:3]
                p['roe_pct'] = (p['unrealised_pnl']) / (p['position_margin'] or 1)
                p['pnl_pct'] = p['roe_pct'] / p['leverage']

                # FIXME need symbol precision!! default rounding down to 0.5 currently
                p['last_price'] = f.get_price(p['pnl_pct'], p['entry_price'], p['side']) if p['side'] else None

                positions.append(p)

        return positions

    def _get_instrument(self, **kw) -> dict:
        """Get symbol info eg last_price"""
        return self.req('Market.symbolInfo', **kw)[0]

    def proc_raw_spec(self, order_spec: Union[dict, List[dict]]) -> Union[dict, List[dict]]:
        """Handle all bybits inconsistent return values

        Parameters
        ----------
        order_spec : Union[dict, List[dict]]
            single or list of order spec dicts

        Returns
        -------
        Union[dict, List[dict]]
            processed order spec dict/list
        """
        def _proc(o: dict):
            try:
                # holy FK bybit didn't even completely change bitmex's spec
                o['order_id'] = o.pop('clOrdID', o.get('order_id', None))
                o['currency'] = o['symbol'][-3:]
                o['exec_inst'] = ''

                if 'stop' in o.get('order_link_id', ''):
                    o['stop_order_type'] = 'Stop'

                # stop order keys
                for k in ('status', 'type', 'id'):
                    k = f'order_{k}'
                    o[k] = o.pop(f'stop_{k}', o.get(k, None))

                # conditional_submit doesn't return order_status
                if not 'order_status' in o:
                    o['order_status'] = 'New'

                # prices can be '0.0' or other price string
                for k in ('stop_loss', 'take_profit', 'stop_px'):
                    if k in o:
                        if o[k] == '' or float(o[k]) == 0:
                            o[k] = None
                        else:
                            o[k] = float(o[k])

                # price strings to float
                for k in ('price', 'base_price'):
                    o[k] = float(o.get(k, 0))

                # qty strings to int
                for k in ('qty', 'cum_exec_qty'):
                    o[k] = float(o.get(k, 0))

                # usdt orders use different key
                if 'created_time' in o.keys():
                    o['created_at'] = o.pop('created_time')
                    o['updated_at'] = o.pop('updated_time')

                # decimals returned are inconsistent length
                for k in ('created_at', 'updated_at'):
                    t = o[k].split('.')[0].replace('Z', '') + 'Z'
                    o[k] = dt.strptime(t, '%Y-%m-%dT%H:%M:%S%z')

                return o
            except Exception as e:
                log.warning(o)
                raise e

        if isinstance(order_spec, list):
            return [_proc(o) for o in order_spec]
        else:
            return _proc(order_spec)

    def _route_order_request(
            self,
            action: str,
            order_specs: List[dict],
            *args, **kw) -> list[dict]:
        """Route order request to bybit
        - handle converting all keys before send + return values
        - reduce_only orders must be submitted min $1 "inside" of open order, else amending will fail

        Parameters
        ----------
        action : str
            submit | amend | cancel | cancel_all
        order_specs : List[dict]
            single or multiple order spec dicts

        Returns
        -------
        list[dict]
            list of returned order specs
        """

        # inspect function to get allowed parameters
        # eg client.Order.Order_new.operation.params
        endpoint = self.order_endpoints[action].split('.')[1]
        cancel_all = action == 'cancel_all'

        # for canceling all, need to make two calls to close stops too...
        if cancel_all:
            order_specs.append(order_specs[0] | dict(order_type='stop'))

        return_specs = []
        # order reduce_only orders first
        # for spec in sorted(order_specs, key=lambda x: not x.get('reduce_only', False)):
        for spec in self.sort_close_orders(order_specs):
            # stupid stop orders have completely different endpoint
            is_stop = spec.get('order_type', '').lower() == 'stop'

            # cancel_orders sends {'symbol': None}
            prefix = 'Linear' if str(spec.get('symbol', '')).lower().endswith('usdt') else ''

            if is_stop:
                # stop order type needs to be market
                if not cancel_all:
                    spec |= dict(
                        order_type='Market',
                        base_price=str(spec['stop_px'] + (1 * (-1 if spec['side'] == 'Buy' else 1))),
                        stop_px=str(spec['stop_px']),
                        trigger_by='IndexPrice')

                if action in ('amend', 'cancel'):
                    spec['stop_order_id'] = spec['order_id']

                base = f'{prefix}Conditional'  # eg Linearconditional
            else:
                base = f'{prefix}Order'

            # get correct endpoint
            endpoint_full = f'{base}.{endpoint}'  # eg Linearconditional.new
            endpoint_query = f'{base}.query'

            if cancel_all and spec['symbol'] is None:
                raise RuntimeError('cancel_all: symbol cannot be None!')

            # bybit forces use of abs contracts + side = 'Sell'
            for key in ('qty', 'p_r_qty'):
                if key in spec.keys():
                    # TODO not forcing str anymore linear
                    if prefix == 'Linear':
                        spec[key] = abs(float(spec[key]))
                    else:
                        spec[key] = int(abs(float(spec[key])))

            # msg = f'[{action}]:\n\t{spec}'
            # log.info(msg)
            ret_spec = self.req(endpoint_full, **spec)

            # check for ret_code errors
            if not ret_spec is None:
                # log.warning(ret_spec)

                # LinearOrder.cancel_all returns ['<order_id>'], reglar returns full spec
                if not (prefix == 'Linear' and action == 'cancel_all'):

                    # NOTE not sure if this is ever needed other than manual testing
                    if action in ('amend', 'cancel'):
                        # kinda annoying but have to call for order again
                        id_key = 'order_id' if not is_stop else 'stop_order_id'

                        # if prefix == 'Linear':
                        #     order_id =

                        ret_spec = self.req(
                            endpoint_query,
                            symbol=spec['symbol'],
                            **{id_key: ret_spec[id_key]})

                    jf.safe_append(return_specs, self.proc_raw_spec(ret_spec))

        return return_specs

    def get_candles(
            self,
            starttime: dt,
            symbol: Union[str, List[str]] = SYMBOl,
            interval: int = 15,
            endtime: dt = None,
            max_pages: float = float('inf'),
            **kw) -> pd.DataFrame:
        """Get OHLC candles from Bybit
        - bybit returns last candle partial (always filter if time == now)
        - max 200 cdls per call
        - bybit doesn't seem to have a ratelimit for ohlc data?

        Parameters
        ----------
        symbol : Union[str, List[str]], optional
            single or list of symbols, default 'BTCUSD'
        interval : int, optional
            default 15
        starttime : dt
        endtime : dt, optional
            default None

        Returns
        -------
        pd.DataFrame
        """
        # save orig startime for per-symbol calls
        _starttime = starttime
        endtime = min(endtime or dt.utcnow(), f.inter_now(interval)).replace(tzinfo=tz.utc)
        data = []
        pages = 0
        _interval = {1: '60', 15: '15'}[interval]
        limit = 200  # max 200 candles for query_kline

        for symbol in jf.as_list(symbol):
            starttime = _starttime.replace(tzinfo=tz.utc)

            # USDT contracts have different endpoint
            prefix = 'Linear' if symbol.lower().endswith('usdt') else ''

            while starttime < endtime and pages < max_pages:
                pages += 1

                try:
                    _data = self.req(
                        f'{prefix}Kline.get',
                        symbol=symbol,
                        interval=_interval,
                        limit=limit,
                        **{'from': starttime.timestamp()})

                    if not _data:
                        break

                    data.extend(_data)
                    starttime = dt.fromtimestamp(_data[-1]['open_time'], tz.utc) + f.inter_offset(interval)
                    log.info(f'{symbol}: starttime={starttime}, len_data={len(_data)}')
                except Exception as e:
                    log.warning(e)
                    log.info(f'starttime: {starttime}, len_data: {len(data)}')
                    break

        cols = ['interval', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

        dtypes = {c: float for c in cols} \
            | dict(interval=int, symbol=str, volume=int)

        if not data:
            log.warning('No candle data returned from bybit.')
            return

        return pd.DataFrame(data) \
            .pipe(pu.safe_drop, 'turnover') \
            .rename(columns=dict(open_time='timestamp')) \
            .astype(dtypes) \
            .assign(timestamp=lambda x: pd.to_datetime(x.timestamp, unit='s')) \
            .pipe(lambda df: df[df.timestamp < f.inter_now(interval)])[cols]

    def get_api_info(self) -> dict:
        keys = self.req('APIkey.info')
        for m in keys:
            for k in ('created_at', 'expired_at'):
                m[k] = dt.strptime(m[k].split('.')[0], '%Y-%m-%dT%H:%M:%S')

        return keys

    def update_api_keys(self, key: str, secret: str) -> None:
        """Update api key and secret in database
        - bybit api keys expire every 3 months

        Parameters
        ----------
        key : str
        secret : str
        """
        from jambot.database import db
        sql = f"update apikeys set [key]='{key}', [secret]='{secret}' where [exchange]='bybit' and [user]='{self.user}'"
        cursor = db.cursor
        cursor.execute(sql)
        cursor.commit()
        log.info(f'Updated exchange key/secret to: key="{key}", secret="{secret}"')

    def check_api_expiry(self, discord_user) -> None:
        """Sort apikeys by expiry date"""
        keys = self.get_api_info()
        m_new = sorted(keys, key=lambda m: m['expired_at'], reverse=True)[0]

        # check if api key expires soon and send message
        if m_new['expired_at'] < dt.utcnow() + delta(days=-3):
            msg = ''
            cm.discord(msg=msg, channel='alerts')

    def list_symbols(self) -> List[str]:
        """Get list of all exchange symbols"""
        lst = self.req('Market.symbolInfo')  # type: List[Dict[str, Any]]
        syms = [m['symbol'] for m in lst]
        return sorted(syms)


def test_candle_availability(interval: int = 5):
    """Run this to test how many seconds till new candle becomes available
    - for testing api latency
    """
    import time

    # starttime must be multiple of interval
    bb = Bybit.default(refresh=False)
    d = dt.utcnow()
    starttime = dt(d.year, d.month, d.day, d.hour, 0)
    # interval = 5
    # interval = 15
    mins = ((dt.utcnow() - starttime).seconds / 60) // 1
    n = int((mins / interval) // 1) + 1
    log.info(f'target candles: {n}')

    nrows = 0
    i = 0
    max_calls = 30

    while nrows < n and i < max_calls:
        time.sleep(1)
        df = bb.get_candles(interval=interval, starttime=starttime)
        nrows = df.shape[0]
        log.info(f'{i:02}: second: {dt.now().second:02}, candles: {nrows}')
        i += 1

    return df
