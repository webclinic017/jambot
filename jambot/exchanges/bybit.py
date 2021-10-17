from datetime import datetime as dt
from datetime import timezone as tz
from typing import *

import pandas as pd
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
        cur_qty='currentQty',
    )

    order_keys = ExchOrder._m_conv.get('bybit')
    api_host = 'https://api.bybit.com'
    api_host_test = 'https://api-testnet.bybit.com'
    api_spec = '/doc/swagger/v_0_2_12.txt'

    order_params = dict(
        submit=dict(func='Order_new'),
        amend=dict(func='Order_replace'),
        cancel=dict(func='Order_cancel'),
        cancel_all=dict(func='Order_cancelAll')
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
        return self.check_request(self.client.Wallet.Wallet_getBalance(coin='BTC'))['result']['BTC']

    def set_orders(self, symbol: str = SYMBOl):
        """set raw order dicts from exch
        limit - max = 50, default 20

        Parameters
        ----------
        symbol : str, optional
            default BTCUSD
        """
        # order_status='New'

        orders = self.client.Order.Order_getOrders(symbol=SYMBOl).result()[0]['result']['data']

        self._orders = self.add_custom_specs(orders)

    def _get_positions(self) -> List[dict]:
        m_raw = self.client.Positions.Positions_myPosition().result()[0]['result']
        return [m['data'] for m in m_raw]

    def _get_instrument(self, **kw) -> dict:
        """"""
        return self.client.Market.Market_symbolInfo(**kw).result()[0]['result'][0]

    def _route_order_request(
            self,
            # func: Callable,
            action: str,
            order_specs: Union[List[ExchOrder], List[dict]],
            *args, **kw):
        # TODO probably need to check all bybit order statuses eg "Created"

        # inspect function to get allowed parameters
        # eg client.Order.Order_new.operation.params
        params = self.order_params.get(action)
        func = getattr(self.client.Order, params['func'])
        func_params = func.operation.params

        return_specs = []
        for order_spec in order_specs:

            spec = {k: v for k, v in order_spec.items() if k in func_params.keys()}

            print(spec)

            # print('spec_submit: ', spec)
            res = self.check_request(func(**spec))
            # print(res)

            # check for ret_code errors
            if not res['ret_code'] == 0:
                log.error(f'Order request failed: \n\t{res}\n\t{spec}')

            # print(res['result'])
            return_spec = res['result']
            if not return_spec is None:
                return_specs.append(return_spec)

        return return_specs

        # ['Order_cancel',
        # 'Order_cancelAll',
        # 'Order_getOrders',
        # 'Order_new',
        # 'Order_query',
        # 'Order_replace']

    def _submit_orders(self, order_specs: list):

        for order_spec in order_specs:
            result = self.client.Order.Order_new
        return

    def get_candles(
            self,
            symbol: str = SYMBOl,
            interval: int = 15,
            starttime: dt = None,
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
        res = self.client.Kline.Kline_get(
            symbol=symbol,
            interval=str(interval),
            **{'from': starttime.replace(tzinfo=tz.utc).timestamp()}).result()[0]

        data = res['result']
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
