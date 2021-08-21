from __future__ import annotations

import uuid
from abc import ABCMeta

from jambot import functions as f
from jambot.common import DictRepr, Serializable
from jambot.tradesys.base import Observer, SignalEvent
from jambot.tradesys.enums import OrderStatus, OrderType, TradeSide

from .__init__ import *

log = getlog(__name__)


class BaseOrder(object, metaclass=ABCMeta):
    """Base class to be inherited by backtest or live exchange orders"""
    ts_format = '%Y-%m-%d %H:%M'

    def __init__(
            self,
            qty: int,
            price: float = None,
            offset: float = None,
            symbol: str = SYMBOL,
            order_id: str = None,
            name: str = '',
            **kw):

        price_original = price
        timestamp_filled = None

        # if pd.isna(qty) or qty == 0:
        #     raise ValueError(f'Order quantity cannot be {qty}')

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
    def side_opp(self):
        """Return qty positive or negative"""
        return self.side * -1

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
    def is_reduce(self) -> bool:
        """If order reduces/closes open position size"""
        return 'close' in self.name

    @property
    def is_increase(self) -> bool:
        """If order increases/opens position"""
        return 'open' in self.name

    @property
    def is_buy(self) -> bool:
        """If order is buy"""
        return self.side == TradeSide.LONG

    @property
    def is_sell(self) -> bool:
        """If order is sell"""
        return self.side == TradeSide.SHORT

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

    @property
    def key_base(self) -> str:
        """Create key base as unique id (per trade)

        Returns
        -------
        str
            formatted key:
            - SYMBOL-name-side_str
            - XBTUSD-limitopen-long
        """
        # if ordtype == 'Stop':
        #     side *= -1

        sidestr = 'long' if self.side == 1 else 'short'
        return '{}-{}-{}'.format(self.symbol, self.name.lower(), sidestr)

    def dict_stats(self) -> dict:

        return dict(
            ts=self.timestamp,
            ts_filled=self.timestamp_filled,
            symbol=self.symbol,
            order_type=self.order_type,
            qty=self.qty,
            price=self.price,
            status=self.status,
            name=self.name)

    def format_ts(self, ts: dt) -> Union[str, None]:
        """Format timestamp as str for dict repr

        Parameters
        ----------
        ts : dt
            timestamp to format

        Returns
        -------
        Union[str, None]
            ts or None
        """
        return ts.strftime(self.ts_format) if not ts is None else None

    def to_dict(self) -> dict:

        price = f'{self.price:,.0f}' if not self.price is None else None

        return dict(
            ts=self.format_ts(self.timestamp_start),
            # ts_filled=self.format_ts(self.timestamp_filled),
            symbol=self.symbol,
            order_type=str(self.order_type),
            qty=f'{self.qty:+,.0f}',
            price=price,
            status=self.status,
            name=self.name)

    def as_bitmex(self) -> 'BitmexOrder':
        """Convert to BitmexOrder"""
        return BitmexOrder.from_base_order(order=self)


class BitmexOrder(BaseOrder, DictRepr, Serializable):
    """Class to represent bitmex live-trading orders"""
    order_type = ''
    ts_format = '%Y-%m-%d %H:%M:%S'

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
        timestamp_start='transactTime',
        name='name')

    def __init__(
            self,
            order_type: str,
            status: str = None,
            order_spec_raw: dict = None,
            order_id: str = None,
            timestamp: dt = None,
            timestamp_start: dt = None,
            key: str = None,
            stop_px: float = None,
            name: str = '',
            **kw):

        # convert stop_px to price for stop orders
        if not stop_px is None:
            kw['price'] = stop_px

        max_name_len = 13
        if len(name) > max_name_len:
            raise ValueError(f'Order name too long: {len(name)}, {name}. Max: {max_name_len}')

        super().__init__(name=name, **kw)

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

    @classmethod
    def example(cls):
        return cls(order_type='limit', qty=-1000, price=50000, name='example_order')

    @property
    def qty(self):
        """Ensure BitmexOrders always in multiples of 100"""
        return f.round_down(n=abs(self._qty), nearest=100) * self.side

    @qty.setter
    def qty(self, val):
        """Set qty and side based on qty
        - NOTE not dry, but can't redefine getter without setter in same class
        """
        self._qty = int(val)
        self._side = TradeSide(np.sign(val))

    @property
    def exec_inst(self) -> list:
        """Create order type specific exec instructions
        - TODO will probably need to allow adding extra exec_inst specs
        """
        lst = []
        if self.is_limit:
            # prevents order from market filling if wrong side or price
            lst.append('ParticipateDoNotInitiate')

        if self.is_stop:
            lst.append('IndexPrice')

        # NOTE kinda sketch, need to make sure name always has "close"
        if self.is_reduce:
            lst.append('Close')

        return lst

    @property
    def exec_inst_str(self) -> str:
        """Get string version of exec_inst to submit with order spec

        Returns
        -------
        str
            values in exec_inst joined
        """
        return ','.join(self.exec_inst)

    def __json__(self) -> dict:
        """Return order spec dict to make json serializeable"""
        return self.order_spec

    @property
    def sym_short(self) -> str:
        """Get symbol's base currency
        - eg remove 'USD' from 'XBTUSD'
        """
        return self.symbol.replace(self.raw_spec('currency'), '')

    @property
    def order_spec(self) -> dict:
        """Create order spec dict to submit to bitmex
        """
        m = dict(
            order_id=self.order_id,
            symbol=self.symbol.upper(),
            order_type=str(self.order_type).title(),
            qty=self.qty,
            key=self.key,
            exec_inst=self.exec_inst_str)

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
        m = {k: v for k, v in self.order_spec.items() if k in keys}

        # check order has an order_id set before it can be amended
        order_id = m.get('orderID', None)
        if order_id is None:
            raise AttributeError(f'orderID not set, needed to amend. orderID: {order_id}')

        return m

    def amend_from_order(self, order: BitmexOrder) -> None:
        """Update self.price and self.qty from different order

        Parameters
        ----------
        order : BitmexOrder
            order to copy data from
        """
        self.price = order.price
        self.qty = order.qty

    def raw_spec(self, key: str) -> Any:
        """Return val from raw order spec

        Parameters
        ----------
        key : str
            dict key to find

        Returns
        -------
        Any
            any value from raw order spec dict

        Raises
        ------
        AttributeError
            if key doesn't exist
        """
        if self.order_spec_raw:
            try:
                return self.order_spec_raw[key]
            except KeyError:
                log.warning(f'key "{key}" doesn\'t exist in order_spec_raw')
                return None
        else:
            raise AttributeError('order_spec_raw not set.')

    @property
    def short_stats(self) -> str:
        """Return compressed version of info for messages"""
        return f'{self.symbol} | {self.name} | ${self.price:,.0f} | {self.qty:+,}'

    def summary_msg(self, exch=None, nearest: float = 0.5) -> str:
        """Get buy/sell price qty summary for discord msg
        -eg "XBT | Sell -2,000 at $44,975.5 (44972.0) | limit_open"

        Returns
        -------
        str
        """
        m = self.order_spec_raw
        avgpx = f.round_down(n=m['avgPx'], nearest=nearest)

        ordprice = f' (${self.price:,})' if not self.price == avgpx else ''

        stats = ''
        if not exch is None:
            stats = f' | Bal: {exch.total_balance_margin:.3f} | ' \
                + f'PnL: {exch.prev_pnl:.3f}' if self.is_stop or self.is_reduce else ''

        return '{} | {:<4} {:>+8,} at ${:,}{:>12} | {}{}'.format(
            self.sym_short,
            m['sideStr'],
            self.qty,
            avgpx,
            ordprice,
            self.name,
            stats)

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
        """Timestamp to make key unique (bitmex rejects duplicate clOrdIds)
        - This is created once, first time key_ts is called
        """
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
        """Create key for clOrdID from params
        - orders will have key in format "SYMBOL-name-side_str-0123456789"
        """
        return '{}-{}'.format(
            self.key_base,
            self.key_ts)

    @key.setter
    def key(self, val):
        """Only set key_ts, get from clOrdID
        - symbol comes from raw ord spec
        - name set as new key when data ingested from raw order spec
        - side_str determined from side
        """
        if not val is None:
            match = re.search(r'\d{10}', val)
            if match:
                self.key_ts = match.group()

    def to_dict(self) -> dict:
        """Add key for BitmexOrders"""
        return super().to_dict() | dict(key=self.key)


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
        amended = SignalEvent()
        timedout = SignalEvent(object)

        # give order unique id
        if order_id is None:
            order_id = str(uuid.uuid1())

        status = OrderStatus.PENDING

        f.set_self(vars())

    @property
    def is_expired(self) -> bool:
        """Check if order has reached its timeout duration"""
        return self.duration >= self.timeout

    def step(self):
        pass
        # if self.is_expired:
        #     print('TIMED OUT')
        #     self.timedout.emit(self)

    def add(self, trade) -> 'Order':
        """Convenience func to add self to trade

        Parameters
        ----------
        trade : Trade

        Returns
        -------
        Order
            self
        """
        if not trade is None:
            trade.add_order(self)

        return self

    def fill(self) -> None:
        """Decide if adding or subtracting qty"""
        self.timestamp_filled = self.timestamp
        self.status = OrderStatus.FILLED
        self.filled.emit(self.qty)  # NOTE not sure if qty needed
        self.detach_from_parent()

    def cancel(self) -> None:
        """Cancel order"""
        self.cancelled.emit()
        self.status = OrderStatus.CANCELLED
        self.detach_from_parent()

    def stoppx(self):
        return self.price * (1 + self.slippage * self.side)

    def adjust_max_qty(self) -> None:
        """Set qty to max available"""
        qty = self.parent.wallet.available_quantity(price=self.price) * self.side
        self.parent.broker.amend_order(order=self, qty=qty)

    def to_dict(self) -> dict:
        """Add t_num for strat Orders"""
        return super().to_dict() | dict(
            ts_filled=self.format_ts(self.timestamp_filled),
            t_num=self.parent.trade_num)


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

    @classmethod
    def from_order(cls, order: Order, stop_pct: float) -> StopOrder:
        stop_price = f.get_price(
            pnl=abs(stop_pct) * -1,
            entry_price=order.price,
            side=order.side)

        stop_order = StopOrder(
            symbol=order.symbol,
            qty=order.qty * -1,
            price=stop_price,
            name='stop_close')

        return stop_order


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


def make_orders(
        order_specs: Union[List[dict], dict],
        as_bitmex: bool = False, **kw) -> Union[List[Order], List[BitmexOrder]]:
    """Make multiple orders

    Parameters
    ----------
    order_specs : Union[List[dict], dict]
        list of order_specs dicts
    as_bitmex : bool
        convert to bitmex orders or not

    Returns
    -------
    List[Order] | List[BitmexOrder]
        list of initialized Order | BitmexOrder objects
    """
    order_specs = f.as_list(order_specs)

    if not as_bitmex:
        orders = [make_order(**order_spec, **kw) for order_spec in order_specs]

    else:
        # dont want to mix up strat Orders with BitmexOrders (strat creates its own order_id)
        orders = [BitmexOrder(**order_spec, **kw) for order_spec in order_specs]

    return orders


def make_bitmex_orders(order_specs: Union[List[dict], dict]) -> List[BitmexOrder]:
    """Create multiple bitmex orders from raw bitmex order spec dicts

    Parameters
    ----------
    order_specs : list
        list of dicts, MUST be response from bitmex

    Returns
    -------
    List[BitmexOrder]
    """
    return [BitmexOrder.from_dict(order_spec) for order_spec in f.as_list(order_specs)]


def list_to_dict(
        orders: List[Union[Order, BitmexOrder]],
        key_base: bool = True) -> Dict[str, Union[Order, BitmexOrder]]:
    """Convenience func to convert list of orders to dict for convenient matching

    Parameters
    ----------
    orders : List[Order | BitmexOrder]
        list of Orders or BitmexOrders
    key_base : bool, optional default True
        use key_base or full key with timestamp as key

    Returns
    -------
    Dict[str, Order | BitmexOrder]
        dict of {order.key_base: order}
    """
    key = 'key_base' if key_base else 'key'
    return {getattr(o, key): o for o in orders}
