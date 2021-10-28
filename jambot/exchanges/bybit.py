from datetime import datetime as dt
from datetime import timedelta as delta
from datetime import timezone as tz
from typing import *

import pandas as pd
from bravado.client import CallableOperation
from bravado.http_future import HttpFuture
from bybit import bybit
from BybitAuthenticator import APIKeyAuthenticator

from jambot import comm as cm
from jambot import functions as f
from jambot import getlog
from jambot.exchanges.exchange import SwaggerExchange
from jambot.tradesys.orders import ExchOrder

SYMBOl = 'BTCUSD'

log = getlog(__name__)

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
    - amend/cancel only return order_id (need to query again) - but cancel_all returns all fields
    - stops have completely different and inconsistent endpoint than market/limit
    - update/created_at times return inconsistent length microsec decimals (or none)
    - stop orders have "stop_order_id" instead of just "order_id"
    - Conditional_query doesnt have any fields to be able to determine its conditional
    - no way to filter orders by filled_time
"""


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

    order_keys = ExchOrder._m_conv.get('bybit')
    api_host = 'https://api.bybit.com'
    api_host_test = 'https://api-testnet.bybit.com'
    api_spec = '/doc/swagger/v_0_2_12.txt'

    order_params = dict(
        submit=dict(func='Order.new'),
        amend=dict(func='Order.replace'),
        cancel=dict(func='Order.cancel'),
        cancel_all=dict(func='Order.cancelAll')
    )

    def __init__(self, user: str, test: bool = False, refresh: bool = False, **kw):
        super().__init__(user=user, test=test, **kw)

        if refresh:
            self.refresh()

    @staticmethod
    def client_cls():
        return bybit

    @staticmethod
    def client_api_auth():
        return APIKeyAuthenticator

    def _get_total_balance(self) -> dict:
        return self.req('Wallet.getBalance', coin='BTC')['BTC']

    def _get_endpoint(self, request: str) -> CallableOperation:
        """Convert simple "Endpoint.request" to actual request
        - eg self.client.Conditional.Conditional_getOrders(symbol=symbol)
        - NOTE may be too simplified but works for now

        Parameters
        ----------
        request : str
            Endpoint.request which represents Endpoint.Endpoint_request()

        Returns
        -------
        Any
            CallableOperation

        Raises
        ------
        RuntimeError
            if request string not in correct format
        """
        if not '.' in request:
            raise RuntimeError(f'Need to have Endpoint.Endpoint_request, not: {request}')

        base, endpoint = request.split('.')
        return getattr(getattr(self.client, base), f'{base}_{endpoint}')

    def req(
            self,
            request: Union[HttpFuture, str],
            code: bool = False,
            fail_msg: str = None,
            **kw) -> Union[Any, Tuple[Any, int]]:
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
            func = self._get_endpoint(request)

            # filter correct keys to submit
            kw = {k: v for k, v in kw.items() if k in func.operation.params.keys()}
            request = func(**kw)

        full_result = self.check_request(request)
        ret_code = full_result['ret_code']

        if not ret_code == 0:
            fail_msg = f'{fail_msg}\n' if not fail_msg is None else ''
            cm.send_error(f'{fail_msg}Request failed:\n\t{full_result}\n\tkw: {kw}', _log=log)

        result = full_result['result']

        if isinstance(result, dict) and 'data' in result:
            result = result['data']

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

        kw = dict(symbol=symbol, limit=30)
        orders = self.req('Order.getOrders', **kw)

        if bybit_async:
            orders += _filter_keys(orders, self.req('Order.query', **kw))

        # very sketch, bybit returns stop orders as "Market", with no price
        orders = [o for o in orders if not 'stop' in o['order_link_id']]

        stop_orders = []
        if bybit_stops:
            stop_orders = self.req('Conditional.getOrders', **kw)

            if bybit_async:
                stop_orders += _filter_keys(stop_orders, self.req('Conditional.query', **kw))

        self._orders = self.add_custom_specs(self.proc_raw_spec(orders + stop_orders))
        # log.warning(f'Setting Orders:\n\t{self._orders}')

    def _set_positions(self) -> List[dict]:
        """Get position info, eg current qty

        Returns
        -------
        List[dict]
            dicts of all positions
        """
        positions = []

        for m in self.req('Positions.myPosition'):
            p = {k: f.str_to_num(v) for k, v in m['data'].items()}
            p['side_str'] = p['side']
            p['side'] = {'Buy': 1, 'Sell': -1}.get(p['side'])
            p['size'] = p['size'] * (p['side'] or 1)
            p['sym_short'] = p['symbol'][:3]
            p['roe_pct'] = (p['unrealised_pnl']) / (p['position_margin'] or 1)
            p['pnl_pct'] = p['roe_pct'] / p['leverage']
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
                    o[k] = int(o.get(k, 0))

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
            order_specs: Union[List[ExchOrder], List[dict]],
            *args, **kw) -> list[dict]:
        """Route order request to bybit
        - handle converting all keys before send + return values
        - reduce_only orders must be submitted min $1 "inside" of open order, else amending will fail

        Parameters
        ----------
        action : str
            submit | amend | cancel | cancel_all
        order_specs : Union[List[ExchOrder], List[dict]]
            single or multiple order spec dicts

        Returns
        -------
        list[dict]
            list of returned order specs
        """

        # inspect function to get allowed parameters
        # eg client.Order.Order_new.operation.params
        params = self.order_params.get(action)
        cancel_all = action == 'cancel_all'
        endpoint = params['func'].split('.')[1]

        # for canceling all, need to make two calls to close stops too...
        if cancel_all:
            order_specs.append(order_specs[0] | dict(order_type='stop'))

        return_specs = []
        # order reduce_only orders first
        for spec in sorted(order_specs, key=lambda x: not x.get('reduce_only', False)):
            # stupid stop orders have completely different endpoint
            is_stop = spec.get('order_type', '').lower() == 'stop'

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

                base = 'Conditional'
            else:
                base = 'Order'

            # get correct endpoint
            endpoint_full = f'{base}.{endpoint}'
            endpoint_query = f'{base}.query'

            if cancel_all and spec['symbol'] is None:
                spec['symbol'] = self.default_symbol

            # bybit forces use of abs contracts + side = 'Sell'
            for key in ('qty', 'p_r_qty'):
                if key in spec.keys():
                    spec[key] = str(abs(int(spec[key])))

            fail_msg = f'[{action}]:\n\t{spec}'
            # log.info(fail_msg)
            ret_spec = self.req(endpoint_full, fail_msg=fail_msg, **spec)

            # check for ret_code errors
            if not ret_spec is None:
                # NOTE not sure if this is ever needed other than manual testing
                if action in ('amend', 'cancel'):
                    # kinda annoying but have to call for order again
                    id_key = 'order_id' if not is_stop else 'stop_order_id'
                    ret_spec = self.req(
                        endpoint_query,
                        symbol=spec['symbol'],
                        **{id_key: ret_spec[id_key]})

                f.safe_append(return_specs, self.proc_raw_spec(ret_spec))

        return return_specs

    def get_candles(
            self,
            starttime: dt,
            symbol: Union[str, List[str]] = SYMBOl,
            interval: int = 15,
            endtime: dt = None,
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
        _interval = {1: '60', 15: '15'}.get(interval)

        for symbol in f.as_list(symbol):
            limit = 200
            starttime = _starttime.replace(tzinfo=tz.utc)

            while starttime < endtime:

                try:
                    _data = self.req(
                        'Kline.get',
                        symbol=symbol,
                        interval=_interval,
                        limit=limit,
                        **{'from': starttime.timestamp()})

                    if not _data:
                        break

                    data.extend(_data)
                    starttime = dt.fromtimestamp(_data[-1]['open_time'], tz.utc) + f.inter_offset(interval)
                    # log.info(f'starttime: {starttime}, len_data: {len(_data)}')
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
            .pipe(f.safe_drop, 'turnover') \
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

    def check_api_expiry(self, discord_user) -> None:
        """Sort apikeys by expiry date"""
        keys = self.get_api_info()
        m_new = sorted(keys, key=lambda m: m['expired_at'], reverse=True)[0]

        # check if api key expires soon and send message
        if m_new['expired_at'] < dt.utcnow() + delta(days=-3):
            msg = ''
            cm.discord(msg=msg, channel='alerts')


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
