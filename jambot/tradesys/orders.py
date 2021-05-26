import uuid
from abc import ABCMeta

from ..common import DictRepr, Serializable
from .__init__ import *
from .base import Observer, SignalEvent
from .enums import OrderStatus, OrderType, TradeSide

log = getlog(__name__)


class BaseOrder(object, metaclass=ABCMeta):
    """Base class to be inherited by backtest or live exchange orders"""

    def __init__(
            self,
            qty: int,
            price: float,
            symbol: str = SYMBOL,
            order_id: str = None,
            name: str = '',
            **kw):

        price_original = price

        if pd.isna(qty) or qty == 0:
            raise ValueError(f'Order quantity cannot be {qty}')

        f.set_self(vars())

    @property
    def qty(self):
        return self._qty

    @qty.setter
    def qty(self, val):
        """Set qty and side based on qty"""
        self._qty = int(val)
        self._side = TradeSide(np.sign(val))

    def increase_qty(self, qty: int):
        """Increase absolute order quantity"""
        self.qty += qty * self.side

    @property
    def side(self):
        """Return qty positive or negative"""
        return self._side

    @property
    def is_open(self):
        return self.status == OrderStatus.OPEN

    @property
    def is_filled(self):
        return self.status == OrderStatus.FILLED

    @property
    def is_cancelled(self):
        return self.status == OrderStatus.CANCELLED

    @property
    def is_limit(self) -> bool:
        return self.order_type == OrderType.LIMIT

    @property
    def is_market(self) -> bool:
        return self.order_type == OrderType.MARKET

    @property
    def is_stop(self) -> bool:
        return self.order_type == OrderType.STOP

    @property
    def trigger_switch(self):
        """Switch to determine direction to sort for checking order price"""
        if self.is_limit:
            return 1
        elif self.is_market:
            return None
        elif self.is_stop:
            return -1

    @property
    def trigger_direction(self) -> int:
        return self.side * self.trigger_switch

    @property
    def sort_key(self):
        """Sort key function to sort open orders
        - sort by:
            - MARKET
            - 'lower' then 'upper' orders
            - price
        - This always stores orders in the correct order to be evaluated when checking for fill
        - Used by Broker

        Returns
        -------
        tuple : (check_side, price)
        """
        if not self.is_market:
            return (-1 * self.trigger_direction, -1 * self.price * self.trigger_direction)
        else:
            return (-1, float('-inf'))

    # def rescale_contracts(self, balance, conf=1):
    #     self.qty = int(conf * f.get_contracts(
    #         xbt=balance,
    #         leverage=self.trade.strat.lev,
    #         entry_price=self.price,
    #         side=self.side,
    #         isaltcoin=self.bm.altstatus))

    def dict_stats(self) -> dict:

        return dict(
            # order_id=self.order_id,
            ts=self.timestamp,
            symbol=self.symbol,
            order_type=self.order_type,
            qty=self.qty,
            price=self.price,
            status=self.status,
            name=self.name
        )

    def to_dict(self) -> dict:
        ts = self.timestamp.strftime('%Y-%m-%d %H') if not self.timestamp is None else None

        return dict(
            # order_id=self.order_id,
            ts=ts,
            symbol=self.symbol,
            order_type=str(self.order_type),
            qty=f'{self.qty:+.0f}',
            price=self.price,
            status=self.status,
            name=self.name)

    def as_bitmex(self) -> 'BitmexOrder':
        """Convert to BitmexOrder"""
        return BitmexOrder.from_base_order(order=self)


class BitmexOrder(BaseOrder, DictRepr, Serializable):
    """Class to represent bitmex live-trading orders"""
    order_type = ''

    # dict to convert bitmex keys
    m_conv = dict(
        order_type='ordType',
        status='ordStatus',
        order_id='orderID',
        qty='orderQty',
        price='price',
        stop_px='stopPx',
        symbol='symbol',
        key='clOrdID',
        exec_inst='execInst',
        timestamp='transactTime',
        name='name',
    )

    def __init__(
            self,
            order_type: str,
            status: str = None,
            order_spec_raw: dict = None,
            order_id: str = None,
            timestamp: dt = None,
            key: str = None,
            **kw):
        super().__init__(**kw)

        order_type = OrderType(order_type)
        status = OrderStatus(status or 'new')
        is_bitmex = True

        f.set_self(vars(), exclude=('order_spec',))

    @classmethod
    def from_dict(cls, order_spec: dict) -> 'BitmexOrder':
        """Create order from bitmex order spec dict"""
        m = {k: order_spec.get(cls.m_conv[k]) for k in cls.m_conv}
        return cls(**m, order_spec_raw=order_spec)

    @classmethod
    def from_base_order(cls, order: 'Order') -> 'BitmexOrder':
        """Create bitmex order from base order
        - used to get final/expected orders from strategy and submit/amend
        """

        return cls(
            order_type=order.order_type,
            price=order.price,
            symbol=order.symbol,
            qty=order.qty,
            name=order.name
        )

    @property
    def exec_inst(self):
        # TODO will probably need to allow adding extra exec_inst specs
        lst = []
        if self.is_limit:
            # prevents order from market filling if wrong side or price
            lst.append('ParticipateDoNotInitiate')

        if self.is_stop:
            lst.append('IndexPrice')

        # NOTE kinda sketch
        if 'close' in self.name:
            lst.append('Close')

        return lst

    @property
    def exec_inst_str(self):
        return ','.join(self.exec_inst)

    def __json__(self):
        """Return order spec dict to make json serializeable"""
        return self.order_spec

    @property
    def order_spec(self) -> dict:
        """Create order spec dict to submit to bitmex
        - TODO need to include order_id when ammending order
        - NOTE could make this static only when vals change?
        """
        m = dict(
            order_id=self.order_id,
            symbol=self.symbol.upper(),
            order_type=str(self.order_type).title(),
            qty=self.qty,
            key=self.key,
            exec_inst=self.exec_inst_str
        )

        # market order doesn't have price
        # stop needs stopPx only
        if self.is_limit:
            m['price'] = self.price
        elif self.is_stop:
            m['stop_px'] = self.price

        # convert back to bitmex keys
        return {self.m_conv.get(k, k): v for k, v in m.items() if not v is None}

    @property
    def order_spec_amend(self) -> dict:
        """Subset of order_spec for amending only"""
        keys = ('orderID', 'symbol', 'orderQty', 'price')
        return {k: v for k, v in self.order_spec.items() if k in keys}

    def raw_spec(self, key: str):
        """Return val from raw order spec"""
        if self.order_spec_raw:
            try:
                return self.order_spec_raw[key]
            except KeyError:
                log.warning(f'key "{key}" doesn\'t exist in order_spec_raw')
                return None
        else:
            raise AttributeError('order_spec_raw not set.')

    @property
    def name(self):
        if self._name == '':
            self._name = 'temp'

        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val).lower()

    @property
    def key_ts(self):
        if not hasattr(self, '_key_ts'):
            self._key_ts = int(time.time())

        return self._key_ts

    @key_ts.setter
    def key_ts(self, val):
        if not (val.isnumeric() and len(val) == 10):
            raise ValueError(f'Key timestamp incorrect value: "{val}"')

        self._key_ts = val

    @property
    def key(self):
        """Create key for clOrdID from params"""
        return '{}-{}'.format(
            f.key(self.symbol, self.name, self.side, self.order_type),
            self.key_ts)

    @key.setter
    def key(self, val):
        """Only set key_ts"""
        if not val is None:
            self.key_ts = val.split('-')[-1]


class Order(BaseOrder, Observer, metaclass=ABCMeta):
    """Order base to simulate backtests"""

    def __init__(
            self,
            order_id: str = None,
            timeout: int = float('inf'),
            **kw):

        super().__init__(**kw)
        Observer.__init__(self)

        filled = SignalEvent(int)
        cancelled = SignalEvent()
        # ammended = SignalEvent(float)
        timedout = SignalEvent(object)

        # give order unique id
        if order_id is None:
            order_id = str(uuid.uuid1())

        status = OrderStatus.PENDING

        f.set_self(vars())

    @property
    def is_expired(self):
        """Check if order has reached its timeout duration"""
        return self.duration >= self.timeout

    def step(self):
        """Check if execution price hit and fill"""
        if self.is_expired:
            self.timedout.emit(self)

    def fill(self):
        """Decide if adding or subtracting qty"""
        self.filled_time = self.timestamp
        self.filled.emit(self.qty)  # NOTE not sure if qty needed
        self.status = OrderStatus.FILLED
        self.detach_from_parent()

    def cancel(self):
        """Cancel order"""
        self.cancelled.emit()
        self.status = OrderStatus.CANCELLED
        self.detach_from_parent()

    def stoppx(self):
        return self.price * (1 + self.slippage * self.side)


class LimitOrder(Order):
    """Limit order which can be set at price and wait to be filled"""
    order_type = OrderType.LIMIT

    @classmethod
    def example(cls):
        return cls(qty=-1000, price=8888)


class MarketOrder(Order):
    """Market order which will be executed immediately"""
    order_type = OrderType.MARKET

    def __init__(self, price: float = None, **kw):
        super().__init__(price=price, **kw)


class StopOrder(Order):
    """Stop order which can be used to ender or exit positions"""
    order_type = OrderType.STOP


def make_order(order_type: 'OrderType', **kw) -> Order:
    """Make single order from dict of order_specs

    Parameters
    ----------
    order_type : OrderType

    Returns
    -------
    Order
        Order object
    """
    cls = dict(
        limit=LimitOrder,
        market=MarketOrder,
        stop=StopOrder).get(str(order_type))

    return cls(**kw)


def make_orders(order_specs: list, as_bitmex: bool = False, **kw) -> list:
    """Make multiple orders

    Parameters
    ----------
    order_specs : list
        list of order_specs dicts
    as_bitmex : bool
        convert to bitmex orders or not

    Returns
    -------
    list
        list of initialized Order objects
    """
    orders = [make_order(**order_spec, **kw) for order_spec in order_specs]

    if as_bitmex:
        orders = [o.as_bitmex() for o in orders]

    return orders


def make_bitmex_orders(order_specs: list) -> List[BitmexOrder]:
    """Create multiple bitmex orders from list of dicts

    Parameters
    ----------
    order_specs : list
        list of dicts, usually comes as response from bitmex

    Returns
    -------
    List[BitmexOrder]
    """

    return [BitmexOrder.from_dict(order_spec) for order_spec in order_specs]
