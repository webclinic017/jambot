import itertools
import json
import re
import time
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty, abstractstaticmethod
from collections import defaultdict as dd
from typing import *

import pandas as pd
import requests
from bravado.client import CallableOperation, SwaggerClient
from bravado.requests_client import RequestsClient
from swagger_spec_validator.common import SwaggerValidationWarning

from jambot import comm as cm
from jambot import config as cf
from jambot import display, getlog
from jambot.common import DictRepr
from jambot.config import AZURE_WEB
from jambot.tradesys import orders as ords
from jambot.tradesys.enums import OrderStatus
from jambot.tradesys.exceptions import PositionNotClosedError
from jambot.tradesys.orders import ExchOrder, Order
from jgutils import functions as jf
from jgutils.secrets import SecretsManager

if TYPE_CHECKING:
    from bravado.http_future import HttpFuture

log = getlog(__name__)


class Exchange(DictRepr, metaclass=ABCMeta):
    """Base object to represent an exchange connection"""

    def __init__(
            self,
            user: str,
            test: bool = False,
            pct_balance: float = 1,
            from_local: bool = True,
            swagger_spec: dict = None,
            api_key: str = None,
            api_secret: str = None,
            discord: str = None,
            **kw):

        self.exch_name = self.__class__.__name__.lower()

        # allow passing in key/secret so can load from db not local csv
        if api_key is None:
            from jambot.database import db
            sql = f"select [key], [secret] from apikeys where [user]='{user}' and exchange='{self.exch_name}'"
            api_key, api_secret = db.cursor.execute(sql).fetchall()[0]
            log.warning('Loading exch api data from db')

        self._creds = dict(key=api_key, secret=api_secret)

        self._client = self.init_client(test=test, from_local=from_local, swagger_spec=swagger_spec)

        jf.set_self()

    @classmethod
    def default(cls, test: bool = True, refresh: bool = True, **kw) -> 'Exchange':
        """Create Exchange obj with default name"""
        user = 'jayme' if not test else 'testnet'
        return cls(user=user, test=test, refresh=refresh, **kw)

    @classmethod
    def from_dict(cls, m: dict, **kw) -> 'Exchange':
        """Init from df_users dict row
        - TODO probs have to change pct_balance key eventually

        Parameters
        ----------
        m : dict
            df row as dict

        Returns
        -------
        Exchange
        """
        kw['test'] = True if 'test' in kw['user'] else False
        return cls(
            pct_balance=m['xbt'],
            api_key=m['key'],
            api_secret=m['secret'],
            discord=m['discord'],
            **kw)

    def to_dict(self) -> dict:
        return dict(user=self.user, test=self.test, pct_balance=self.pct_balance)

    @property
    def key(self):
        return self._creds['key']

    @property
    def secret(self):
        return self._creds['secret']

    @abstractmethod
    def init_client(self):
        """Initialize client connection"""
        pass

    @abstractmethod
    def refresh(self):
        """Refresh position/orders/etc from exchange"""
        pass

    @abstractmethod
    def set_orders(self):
        """Load/save recent orders from exchange"""
        pass

    def load_creds(self, user: str):
        """Load creds from csv"""
        df = SecretsManager(f'{self.__class__.__name__.lower()}.csv').load \
            .set_index('user')

        if not user in df.index:
            raise RuntimeError(f'User "{user}" not in saved credentials.')

        return df.loc[user].to_dict()


class SwaggerAPIException(Exception):
    def __init__(
            self,
            request: 'HttpFuture',
            code: int = 400,
            api_message: str = None,
            fail_msg: str = None) -> None:
        """Raise exception on bybit invalid api request

        Parameters
        ----------
        request : HttpFuture
            original request to get operation name
        code : int, optional
            error code, default 400
        api_message : str, optional
            api failure message, default None
        fail_msg : str, optional
            custom additional err info message, default None
        """
        try:
            operation = request.operation.op_spec['operationId']
        except Exception:
            # just in case
            operation = '*Missing Operation*'

        fail_msg = f'\n\t{fail_msg}\n\t' if not fail_msg is None else ''

        request_data = request.future.request.data  # data submitted to api

        msg = f'{code} - {api_message}{fail_msg}\n\t{operation}: {request_data}'

        log.error(msg)
        super().__init__(msg)
        self.msg = msg

    def send_error_discord(self) -> None:
        """Send error info to discord"""
        cm.send_error(self.msg, force=True)


class SwaggerExchange(Exchange, metaclass=ABCMeta):
    """Class for exchanges using Swagger spec"""
    div = abstractproperty()
    default_symbol: str = abstractproperty()
    wallet_keys = abstractproperty()
    order_keys = abstractproperty()
    api_host = abstractproperty()
    api_host_test = abstractproperty()
    api_spec = abstractproperty()
    order_endpoints = abstractproperty()
    conv_symbol = {}
    other_keys = abstractproperty()
    m_emoji = dict(submit='✅ ', amend='🌀 ', cancel='❌ ')

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.balance_set = False
        self.avail_margin = 0
        self._total_balance_margin = None
        self.total_balance_wallet = 0
        self.reserved_balance = 0
        self.unrealized_pnl = 0
        self.prev_pnl = 0
        self._orders = None
        self._positions = None

    @staticmethod
    @abstractstaticmethod
    def client_cls() -> Callable:  # type: ignore
        pass

    @staticmethod
    @abstractstaticmethod
    def client_api_auth() -> Callable:  # type: ignore
        """Bybit/bitmex have similar but slightly different api authenticators"""
        pass

    @property
    def client(self) -> SwaggerClient:
        """Client connection to exchange"""
        return self._client

    @classmethod
    def load_swagger_spec(cls, test: bool) -> dict:
        """Load swagger spec from local file
        - NOTE loading from local only takes ~1ms, not faster to bother passing in per user
        """
        exchange = cls.__name__.lower()
        name = exchange
        if test:
            name += '-test'

        p = cf.p_res / f'swagger/{name}.json'
        with open(p, 'rb') as file:
            return json.load(file)

    @classmethod
    def update_local_spec(cls) -> None:
        """Get updated swagger.json spec from exch api and write to local"""
        hosts = [cls.api_host, cls.api_host_test]
        urls = [host + cls.api_spec for host in hosts]
        exchange = cls.__name__.lower()

        for url in urls:
            name = exchange
            if 'test' in url:
                name += '-test'

            p = cf.p_res / f'swagger/{name}.json'

            with open(p, 'w+', encoding='utf-8') as file:
                m = json.loads(requests.get(url).text)
                json.dump(m, file)

            log.info(f'Saved swagger spec from: {url}')

    def init_client(
            self,
            test: bool = False,
            from_local: bool = True,
            swagger_spec: dict = None) -> SwaggerClient:
        """Init api swagger client

        Parameters
        ----------
        test : bool, optional
            use testnet api, by default False
        from_local : bool, optional
            load locally saved swagger spec instead of from html
            (0.5s load total instead of ~1.3s per user), by default True
            - NOTE not sure if this will work indefinitely, or fail quickly if exchange changes api

        Returns
        -------
        SwaggerClient
            initialized swagger api client
        """
        warnings.simplefilter('ignore', SwaggerValidationWarning)

        if from_local:
            host = self.api_host_test if test else self.api_host
            spec_uri = host + self.api_spec

            config = dict(
                use_models=False,
                validate_responses=False,
                also_return_response=True,
                host=host)

            request_client = RequestsClient()
            request_client.authenticator = self.client_api_auth()(host, self.key, self.secret)

            # either load from file, or pass in for faster loading per exchange user
            spec = swagger_spec or self.load_swagger_spec(test=test)

            return SwaggerClient.from_spec(
                spec_dict=spec,
                origin_url=spec_uri,
                http_client=request_client,
                config=config)

        else:
            return self.client_cls()(test=test, api_key=self.key, api_secret=self.secret)

    def refresh(self):
        """Set margin balance, current position info, all orders"""
        self.set_total_balance()
        self.set_positions()
        self.set_orders()

    @abstractmethod
    def req(self, request: str, **kw) -> Any:
        return

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
            raise RuntimeError(f'Need to use "Endpoint.Endpoint_request", not: {request}')

        base, endpoint = request.split('.')
        return getattr(getattr(self.client, base), f'{base}_{endpoint}')

    def _make_request(self, request: str, **kw) -> 'HttpFuture':
        """Build request from str request and request params (kws)

        Parameters
        ----------
        request : str
            [description]

        Returns
        -------
        Union[Any, List[Any]]
            [description]
        """
        func = self._get_endpoint(request)

        # filter correct keys to submit
        kw = {k: v for k, v in kw.items() if k in func.operation.params.keys()}
        return func(**kw)

    def _reapply_auth(self, request: 'HttpFuture') -> 'HttpFuture':
        """reapply authentication timeout/signature to HttpFuture request
        - bitmex auth = request.future.request.headers['api-expires']
        - bybit auth = request.future.request.params['timeout]

        Parameters
        ----------
        request : HttpFuture
            eg self.client.Position.Position_get()

        Returns
        -------
        HttpFuture
            request with authentication/timeout reset
        """
        request.future.request = request.operation.swagger_spec \
            .http_client.apply_authentication(request.future.request)

        return request

    def check_request(self, request: 'HttpFuture', retries: int = 0) -> Any:
        """
        Perform http request and retry with backoff if failed
        - bitmex raises 400 bravado.exception.HTTPBadRequest on invalid request
        - bybit just returns status codes w response

        - type(request) = bravado.http_future.HttpFuture
        - type(response) = bravado.response.BravadoResponse
        - response = request.response(fallback_result=[400], exceptions_to_catch=bravado.exception.HTTPBadRequest)
        """
        backoff = 0.5
        response = request.response(fallback_result='')
        status = response.metadata.status_code

        if status < 300:
            if self.exch_name == 'bitmex':
                ratelim_remaining = int(response.metadata.headers['x-ratelimit-remaining'])
                # "x-ratelimit-reset": "1635796268"  # could wait for this time

                if ratelim_remaining <= 1:
                    log.warning('Ratelimit reached. Sleeping 10 seconds.')
                    time.sleep(10)

            return response.result
        elif status == 503 and retries < 7:
            retries += 1
            sleeptime = backoff * (2 ** retries - 1)
            time.sleep(sleeptime)

            return self.check_request(request=self._reapply_auth(request), retries=retries)
        else:
            cm.send_error('{}: {}\n{}'.format(status, response.result, request.future.request.data))

    @property
    def total_balance_margin(self) -> float:
        val = self._total_balance_margin
        if val is None:
            log.warning('balance not set, refreshing')
            self.set_total_balance()

        return self._total_balance_margin

    @total_balance_margin.setter
    def total_balance_margin(self, val: float) -> None:
        self._total_balance_margin = val

    @abstractmethod
    def _get_total_balance(self) -> dict:
        """swagger call to get user wallet balances"""
        pass

    def set_total_balance(self) -> None:
        """Set margin/wallet values"""
        div = self.div
        res = self._get_total_balance()

        for name, exch_key in self.wallet_keys.items():
            setattr(self, name, res[exch_key] / div)

        # total available/unused > only used in postOrder "Available Balance"
        # self.avail_margin = res['excessMargin'] / div
        # self.total_balance_margin = res['marginBalance'] / div  # unrealized + realized > wallet total qty avail
        # self.total_balance_wallet = res['walletBalance'] / div  # realized
        # self.unrealized_pnl = res['unrealisedPnl'] / div
        # self.prev_pnl = res['prevRealisedPnl'] / div

        self.balance_set = True
        self.res = res

    @abstractmethod
    def _set_positions(self) -> List[dict]:
        """Get position dicts from exchange"""
        pass

    @property
    def positions(self) -> List[dict]:
        """List of all positions"""
        if self._positions is None:
            self.set_positions()

        return self._positions

    def set_positions(self) -> None:
        """Set position for all symbols"""
        positions = self._set_positions()

        # store as dict with symbol as keys
        self._positions = {}

        # save positions to dict with symbol as key
        for p in positions:
            symbol = p['symbol']
            p |= {k: p.pop(v) for k, v in self.pos_keys.items()}
            self._positions[symbol.lower()] = p

    @property
    def balance(self) -> float:
        """Get current exchange balance in Xbt, minus user-defined "reserved" balance
        - "Wallet Balance" on bitmex, NOT "Available Balance"

        Returns
        -------
        float
        """
        return self.total_balance_wallet - self.reserved_balance

    @property
    def orders(self):
        return self._orders

    def get_position(self, symbol: str, refresh: bool = False) -> dict:
        """Get position for specific symbol

        Parameters
        ----------
        symbol : str
            position data for symbol
        refresh : bool, optional
            force refresh of position data, by default False

        Returns
        -------
        dict
            position data
        """
        if refresh:
            self.set_positions()

        return self.positions.get(symbol.lower(), {})

    @abstractmethod
    def _get_instrument(self, **kw) -> dict:
        pass

    def get_instrument(self, symbol: str = None) -> dict:
        """Get symbol stats dict
        - add precision for rounding
        - Useful for getting precision or last price
        - no rest api delay
        - no caching, this always refreshes live

        Parameters
        ----------
        symbol : str, optional

        Returns
        -------
        dict
        """
        symbol = symbol or self.default_symbol
        m = self._get_instrument(symbol=symbol)

        # NOTE proper ticksize not set for bybit for symbols other than BTCUSD
        return m | dict(precision=len(str(m.get('tickSize', 0.5)).split('.')[-1]))

    def last_price(self, symbol: str = None) -> float:
        """Get last price for symbol (used for testing)
        - used in Broker.expected_orders

        Parameters
        ----------
        symbol : str
            eg XBTUSD

        Returns
        -------
        float
            last price
        """
        symbol = symbol or self.default_symbol
        return float(self.get_instrument(symbol=symbol)[self.other_keys['last_price']])

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
        _qty = self.other_keys['cur_qty']
        if not symbol is None:
            return self.get_position(symbol).get(_qty, 0)
        else:
            # all position qty
            return {k: v[_qty] for k, v in self.positions.items()}

    def current_entry(self, symbol: str) -> float:
        return self.get_position(symbol)['entry_price']

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

        for o in jf.as_list(order_specs):
            if isinstance(o, list):
                log.error(f'order is list: {o}')
                return

            if 'processed' in o.keys():
                raise RuntimeError('Order spec has already been processed!')

            o['processed'] = True
            o['exchange'] = self.exch_name

            o['side_str'] = o['side']
            o['side'] = 1 if o['side'] == 'Buy' else -1

            # some orders can have none qty if "close position" market order cancelled by bitmex
            # NOTE kinda messy, should just convert all raw specs to nice keys by default
            for k in ('qty', 'cum_qty'):
                _qty = self.order_keys[k]
                if not o[_qty] is None:
                    o[_qty] = int(o['side'] * int(o[_qty]))

            # add key to the order, excluding manual orders
            _link_id = self.order_keys['order_link_id']
            if not o.get(_link_id, '') == '':
                o['name'] = o[_link_id].split('-')[1]

                if not 'manual' in o[_link_id]:
                    # replace 10 digit timestamp from key if exists
                    o['key'] = re.sub(r'-\d{10}', '', o[_link_id])
                    o['manual'] = False
                else:
                    o['manual'] = True
            else:
                o['name'] = '(manual)'
                o['manual'] = True

        return order_specs

    def get_orders(
            self,
            symbol: str = None,
            new_only: bool = False,
            # filled_only: bool = False,
            bot_only: bool = False,
            manual_only: bool = False,
            as_exch_order: bool = True,
            as_dict: bool = False,
            refresh: bool = False,
            **kw) -> Union[List[dict], List[ExchOrder], Dict[str, ExchOrder]]:
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
            return ExchOrders instead of raw list of dicts, default True
        as_dict : bool, optional
            return Dict[str, ExchOrders] instead of list (for matching)
        refresh : bool, optional
            run self.set_orders() first to ensure orders current with exchange

        Returns
        -------
        Union[List[dict], List[ExchOrder] Dict[str, ExchOrder]]
            list of RAW order dicts, list of exchange orders, or dict of str: ExchOrder
        """
        if symbol is None:
            symbol = self.default_symbol

        if refresh or self._orders is None:
            # kw = bybit stops/async
            self.set_orders(**kw)

        var = {k: v for k, v in vars().items() if not v in ('as_exch_order', 'refresh', 'as_dict')}

        # define filters
        conds = dict(
            symbol=lambda x: x['symbol'].lower() == symbol.lower(),
            new_only=lambda x: OrderStatus(x[self.order_keys['status']]) == OrderStatus.OPEN,
            # filled_only=lambda x: OrderStatus(x[self.order_keys['status']]) == OrderStatus.FILLED,
            bot_only=lambda x: not x['manual'],
            manual_only=lambda x: x['manual'])

        # filter conditions based on true args in vars()
        conds = {k: v for k, v in conds.items() if not var.get(k) in (None, False)}

        orders = [o for o in self._orders if all(cond(o) for cond in conds.values())]

        if as_exch_order:
            orders = self.exch_order_from_raw(order_specs=orders, process=False)

        if as_dict:
            orders = ords.list_to_dict(orders, use_ts=True)

        return orders

    def df_orders(
            self,
            symbol: str = None,
            new_only: bool = True,
            refresh: bool = False,
            include_id: bool = False) -> pd.DataFrame:
        """Used to display orders in google sheet
        - TODO this is slightly inefficient, ~0.7s.. try speeding up raw spec

        Parameters
        ----------
        symbol : str, optional
            default self.default_symbol
        new_only : bool, optional
            default True
        refresh : bool, optional
            default False

        Returns
        -------
        pd.DataFrame
            df of orders
        """
        symbol = symbol or self.default_symbol
        orders = self.get_orders(symbol=symbol, new_only=new_only, refresh=refresh, as_exch_order=True)
        cols = ['order_type', 'name', 'qty', 'price', 'exec_inst', 'symbol']

        if include_id:
            # viewing locally
            cols = ['timestamp', 'order_id'] + cols
            sort_cols = ['timestamp']
            ascending = [False]
        else:
            # google sheets
            sort_cols = ['symbol', 'order_type', 'name']
            ascending = [False, True, True]

        if not orders:
            df = pd.DataFrame(columns=cols, index=range(1))
        else:
            data = [{k: o.raw_spec(k) for k in cols} for o in orders]
            df = pd.DataFrame.from_dict(data) \

        df = df \
            .reindex(columns=cols) \
            .sort_values(
                by=sort_cols,
                ascending=ascending)

        if include_id:
            df = df.assign(timestamp=lambda x: x.timestamp.dt.tz_localize(None))

        return df

    def exch_order_from_raw(self, order_specs: List[dict], process: bool = True) -> List[ExchOrder]:
        """Create exchange order objs from raw specs
        - top level wrapper to both add raw specs and convert to ExchOrder

        Parameters
        ----------
        order_specs : List[dict]
            list of order specs from exchange

        Returns
        -------
        List[ExchOrder]
            list of exchange orders
        """
        if process:
            order_specs = self.add_custom_specs(order_specs)

        # TODO temp solution should handle depending on result
        # order_specs is None
        if order_specs is None:
            log.warning('order_specs is None.')

        for item in order_specs:
            if not isinstance(item, dict):
                raise AttributeError(
                    f'Invalid order specs returned from {self.exch_name}. {type(item)}: {item}')

        return ords.make_exch_orders(order_specs, exch_name=self.exch_name)

    def convert_exch_keys(self, order_specs: List[Union[ExchOrder, dict]]) -> List[dict]:
        """Convert ExchOrder or dict to exch-specific keys

        Parameters
        ----------
        order_specs : List[Union[ExchOrder, dict]]

        Returns
        -------
        List[dict]
            list of converted order specs
        """
        return [{self.order_keys.get(k, k): v for k, v in spec.items()} for spec in jf.as_list(order_specs)]

    def sort_close_orders(self, order_specs: List[dict]) -> List[dict]:
        """Sort orders to submit by close/reduce_only first
        - to avoid api errors

        Parameters
        ----------
        order_specs : List[dict]
            list of orders specs to sort

        Returns
        -------
        List[dict]
            list of sorted order specs
        """
        return sorted(jf.as_list(order_specs), key=lambda x: not x.get('reduce_only', False))

    @abstractmethod
    def _route_order_request(self):
        """Bybit/Bitmex have to handle bulk orders differently"""
        pass

    def _order_request(
            self,
            action: str,
            order_specs: List[Union[ExchOrder, dict]]) -> Union[List[ExchOrder], None]:
        """Send order submit/amend/cancel request

        - intake ExchageOrders OR dict (eg cancel_all) and pass to exchange specific _route_order_request
        - Exchanges know which keys to use per operation

        Parameters
        ----------
        action : str
            submit | amend | cancel | cancel_all
        order_specs : List[Union[ExchOrder, dict]]
            list of orders to process

        Returns
        -------
        Union[List[ExchOrder], None]
            list of order results or None if request failed
        """
        if not order_specs and not action == 'cancel_all':
            return

        # print('specs before: ', order_specs)
        order_specs = self.convert_exch_keys(order_specs)
        # print('\n\nspecs after: ', order_specs)

        # temp convert XBTUSD <> BTCUSD for bitmex/bybit
        for spec in order_specs:
            for _find, replace in self.conv_symbol.items():
                if spec.get('symbol', None) == _find:
                    spec['symbol'] = replace

        # exchange specific
        # print(action, order_specs)
        result = self._route_order_request(action=action, order_specs=order_specs)

        # result can be None if bad orders
        if result is None:
            return

        resp_orders = self.exch_order_from_raw(order_specs=result)

        # check if submit/amend orders incorrectly cancelled
        if not 'cancel' in action:
            failed_orders = [o for o in resp_orders if o.is_cancelled]

            # TODO currently bitmex specific
            if failed_orders:
                msg = 'ERROR: Order(s) CANCELLED!'
                for o in failed_orders:
                    err_text = o.raw_spec(self.other_keys['err_text'])
                    m = o.order_spec

                    # failed submitted at offside price, add last_price for comparison
                    if 'ParticipateDoNotInitiate' in err_text:
                        m['last_price'] = self.last_price(symbol=o.symbol)

                    msg += f'\n\n{err_text}\n{jf.pretty_dict(m, prnt=False, bold_keys=True)}'

                cm.discord(msg, channel='err')

        return resp_orders

    def amend_orders(self, orders: Union[List[ExchOrder], ExchOrder]) -> List[dict]:
        """Amend order price and/or qty

        Parameters
        ----------
        orders : Union[List[ExchOrder], ExchOrder]
            Orders to amend price/qty

        Returns
        -------
        List[ExchOrder]
        """
        return self._order_request(action='amend', order_specs=orders)

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
        for o in jf.as_list(orders):
            # convert dict specs to ExchOrder
            if not isinstance(o, ExchOrder):
                o = ords.make_exch_orders(order_specs=o, exch_name=self.exch_name)[0]

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

        # NOTE don't necessarily need to batch bybit orders this way since all individual
        result = []
        for order_batch in batches:
            result_orders = self._order_request(action='submit', order_specs=order_batch)
            result.extend(result_orders or [])

        return result

    def close_position(self, symbol: str = None) -> None:
        """Market close current open position
        - only used in testing currently
        - TODO make exch specific close position funcs

        Parameters
        ----------
        symbol : str, optional
            symbol to close position, by default 'XBTUSD'
        """
        symbol = symbol or self.default_symbol
        try:
            if self.exch_name == 'bitmex':
                self.req('Order.new', symbol=symbol, execInst='Close')

            elif self.exch_name == 'bybit':
                # NOTE ugh so messy hopefully a way to make this cleaner
                pos = self.get_position(symbol, refresh=True)

                if not pos['qty'] == 0:
                    close_order = dict(
                        order_type='market',
                        symbol=symbol,
                        qty=pos['qty'] * -1,
                        name='pos_close')
                    self.submit_orders(close_order)

        except:
            cm.send_error(msg='ERROR: Could not close position!', _log=log)
        finally:
            pos = self.get_position(symbol, refresh=True)
            if not pos['qty'] == 0:
                raise PositionNotClosedError(qty=pos['qty'])

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
        # order_specs = [o.order_id for o in jf.as_list(orders)]

        if not orders:
            log.warning('No orders to cancel.')
            return

        return self._order_request(action='cancel', order_specs=orders)

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

    def reconcile_orders(
            self,
            symbol: str,
            expected_orders: List[Order],
            test: bool = False,
            **kw) -> None:
        """Compare expected and actual (current) orders, adjust as required

        Parameters
        ----------
        expected_orders : List[Order]
            orders from current timestamp in strat backtest
        actual_orders : List[Order]
            orders active on exchange
        """
        actual_orders = self.get_orders(symbol=symbol, new_only=True, bot_only=True,
                                        as_exch_order=True, refresh=True, **kw)
        all_orders = self.validate_orders(expected_orders, actual_orders, show=True)

        # perform action reqd for orders except valid/manual
        if not test:
            for action, orders in all_orders.items():
                if orders and not action in ('valid', 'manual'):
                    getattr(self, f'{action}_orders')(orders)

        # temp send order submit details to discord
        user = self.user if self.discord is None else self.discord

        m = {}
        m_ords = {k: [o.short_stats for o in orders]
                  for k, orders in all_orders.items() if orders and not k == 'manual'}

        if m_ords:
            # TODO add avg entry price
            m['current_qty'] = f'{self.current_qty(symbol=symbol):+,} @ ${self.current_entry(symbol):,.0f}'
            m |= m_ords

            # add emoji actions
            m = {f'{self.m_emoji.get(k, "")}{k}': v for k, v in m.items()}

            msg = jf.pretty_dict(m, prnt=False, bold_keys=True)
            cm.discord(msg=f'{user}\n{msg}', channel='orders' if not test else 'test')

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
        expected_orders = ords.list_to_dict(expected_orders, use_ts=False)
        actual_orders = ords.list_to_dict(actual_orders, use_ts=False)
        # log.debug(f'\n\nexpected_orders:\n\t{expected_orders}')
        # log.debug(f'\n\nactual_orders:\n\t{actual_orders}')
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
