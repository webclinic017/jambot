from datetime import datetime as dt
from datetime import timezone as tz
from typing import *

import pandas as pd
from bravado.client import CallableOperation
from bravado.http_future import HttpFuture
from bybit import bybit
from BybitAuthenticator import APIKeyAuthenticator

from jambot import functions as f
from jambot import getlog
from jambot.exchanges.exchange import SwaggerExchange
from jambot.tradesys.orders import ExchOrder

SYMBOl = 'BTCUSD'

log = getlog(__name__)


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
            request = self._get_endpoint(request)(**kw)

        full_result = self.check_request(request)
        ret_code = full_result['ret_code']
        if not ret_code == 0:
            fail_msg = f'{fail_msg}\n' if not fail_msg is None else ''
            f.send_error(f'{fail_msg}Request failed:\n\t{full_result}', _log=log)

        result = full_result['result']

        if isinstance(result, dict) and 'data' in result:
            result = result['data']

        # default, just give the data
        if not code:
            return result
        else:
            return result, ret_code

    def set_orders(self, symbol: str = SYMBOl, bybit_async: bool = False, bybit_stops: bool = True):
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

        orders = self.req('Order.getOrders', symbol=symbol)

        if bybit_async:
            orders += self.req('Order.query', symbol=symbol)

        # very sketch, bybit returns stop orders as "Market", with no price
        orders = [o for o in orders if not 'stop' in o['order_link_id']]

        stop_orders = []
        if bybit_stops:
            stop_orders = self.req('Conditional.getOrders', symbol=symbol)

            if bybit_async:
                stop_orders += self.req('Conditional.query', symbol=symbol)

        self._orders = self.add_custom_specs(self.proc_raw_spec(orders + stop_orders))

    def _get_positions(self) -> List[dict]:
        """Get position info, eg current qty

        Returns
        -------
        List[dict]
            dicts of all positions
        """
        m_raw = self.req('Positions.myPosition')
        positions = [m['data'] for m in m_raw]
        for pos in positions:
            pos['side_str'] = pos['side']
            pos['side'] = {'Buy': 1, 'Sell': -1}.get(pos['side'])
            pos['qty'] = pos['size']

        return positions

    def _get_instrument(self, **kw) -> dict:
        """Get symbol info eg last_price"""
        return self.req('Market.symbolInfo', **kw)[0]

    def proc_raw_spec(self, order_spec: Union[dict, List[dict]]) -> Union[dict, List[dict]]:
        """Handle all bybits inconsistent return values

        Issues
        ------
        - conditional_cancel_all still uses bitmex clOrdID (LOL IDIOTS)
        - conditional_submit doesn't return order_status
        - conditional_submit returns order type, but not stop_order_type
        - inconsistent price/qty return types (sometimes string, sometimes int)
        - amend/cancel only return order_id (need to query again) - but cancel_all returns all fields
        - stops have completely different and inconsistent endpoint than market/limit
        - update/created_at times return inconsistent length microsec decimals (or none)
        - stop orders have "stop_order_id" instead of just "order_id"
        - Conditional_query doesnt have any fields to be able to determine its conditional

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

                if 'stop' in o.get('order_link_id', ''):
                    o['stop_order_type'] = 'Stop'

                # stop order keys
                for k in ('status', 'type', 'id'):
                    k = f'order_{k}'
                    o[k] = o.pop(f'stop_{k}', o.get(k, None))

                # conditional_submit doesn't return order_status
                if o.get('order_status', None) is None:
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
                for k in ('qty', ):
                    o[k] = int(o[k])

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
        for spec in order_specs:
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
            func = self._get_endpoint(endpoint_full)
            func_params = func.operation.params

            # filter correct keys to submit
            spec = {k: v for k, v in spec.items() if k in func_params.keys()}

            if cancel_all and spec['symbol'] is None:
                spec['symbol'] = self.default_symbol

            # bybit forces use of abs contracts + side = 'Sell'
            for key in ('qty', 'p_r_qty'):
                if key in spec.keys():
                    spec[key] = str(abs(int(spec[key])))

            fail_msg = f'[{action}]:\n\t{spec}'
            ret_spec = self.req(endpoint_full, fail_msg=fail_msg, **spec)

            # check for ret_code errors
            if not ret_spec is None:
                if action in ('amend', 'cancel'):
                    # kinda annoying but have to call for order again
                    # NOTE might only need to do this during testing
                    id_key = 'order_id' if not is_stop else 'stop_order_id'
                    ret_spec = self.req(endpoint_query, symbol=spec['symbol'], **{id_key: ret_spec[id_key]})

                f.safe_append(return_specs, self.proc_raw_spec(ret_spec))

        return return_specs

    def get_candles(
            self,
            starttime: dt,
            symbol: str = SYMBOl,
            interval: int = 15,
            include_partial: bool = False) -> pd.DataFrame:
        """Get OHLC candles from Bybit
        - NOTE bybit returns last candle partial

        Parameters
        ----------
        symbol : str, optional
            default SYMBOl
        interval : int, optional
            default 15
        starttime : dt, optional
            default None

        Returns
        -------
        pd.DataFrame
        """
        data = self.req(
            'Kline.get',
            symbol=symbol,
            interval=str(interval),
            **{'from': starttime.replace(tzinfo=tz.utc).timestamp()})

        cols = ['interval', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

        dtypes = {c: float for c in cols} \
            | dict(interval=int, symbol=str, volume=int)

        df = pd.DataFrame(data) \
            .pipe(f.safe_drop, 'turnover') \
            .rename(columns=dict(open_time='timestamp')) \
            .astype(dtypes) \
            .assign(timestamp=lambda x: pd.to_datetime(x.timestamp, unit='s'))[cols]

        if not include_partial:
            df = df.iloc[:-1]

        return df
