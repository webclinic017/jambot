import json
import re
import time
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import *

from bravado.client import SwaggerClient
from swagger_spec_validator.common import SwaggerValidationWarning

from jambot import functions as f
from jambot import getlog
from jambot.config import AZURE_WEB
from jambot.utils.secrets import SecretsManager

log = getlog(__name__)


class Exchange(object, metaclass=ABCMeta):
    """Base object to represent an exchange connection"""

    def __init__(self, user: str, test: bool = False, pct_balance: float = 1, **kw):

        self._creds = self.load_creds(user=user)
        self._client = self.init_client(test=test)

        f.set_self(vars())

    @classmethod
    def default(cls, test: bool = True, refresh: bool = True, **kw) -> 'Exchange':
        """Create Exchange obj with default name"""
        user = 'jayme' if not test else 'testnet'
        return cls(user=user, test=test, refresh=refresh, **kw)

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


class SwaggerExchange(Exchange, metaclass=ABCMeta):
    """Class for exchanges using Swagger spec"""
    div = abstractproperty()
    client_cls = abstractproperty()
    wallet_keys = abstractproperty()
    order_keys = abstractproperty()

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.balance_set = False
        self.avail_margin = 0
        self.total_balance_margin = 0
        self.total_balance_wallet = 0
        self.reserved_balance = 0
        self.unrealized_pnl = 0
        self.prev_pnl = 0
        self._orders = None
        self._positions = None

    def init_client(self, test: bool = False) -> SwaggerClient:
        """Init api swagger client"""
        warnings.simplefilter('ignore', SwaggerValidationWarning)
        return self.client_cls()(test=test, api_key=self.key, api_secret=self.secret)

    def refresh(self):
        """Set margin balance, current position info, all orders"""
        self.set_total_balance()
        self.set_positions()
        self.set_orders()

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
    def _get_positions(self) -> List[dict]:
        """position dicts"""
        pass

    def set_positions(self) -> None:
        """Set position for all symbols"""
        res = self._get_positions()

        # store as dict with symbol as keys
        self._positions = {}

        for pos in res:
            symbol = pos['symbol']
            self._positions[symbol.lower()] = pos

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

    @property
    def orders(self):
        return self._orders

    def get_position(self, symbol: str) -> dict:
        """Get position for specific symbol"""
        return self._positions.get(symbol.lower(), {})

    @property
    def client(self) -> SwaggerClient:
        """Client connection to exchange"""
        return self._client

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
        _qty = self.order_keys['qty']
        _link_id = self.order_keys['order_link_id']

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
            if not o[_qty] is None:
                o[_qty] = int(o['side'] * o[_qty])

            # add key to the order, excluding manual orders
            if not o[_link_id] == '':
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
