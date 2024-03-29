from __future__ import annotations

import copy
import re
import time
import uuid
from abc import ABCMeta
from datetime import datetime as dt
from typing import *

import numpy as np

from jambot import SYMBOL, Num
from jambot import functions as f
from jambot import getlog
from jambot.common import DictRepr, Serializable
from jambot.tradesys.base import Observer, SignalEvent
from jambot.tradesys.enums import OrderStatus, OrderType, TradeSide
from jambot.tradesys.symbols import Symbol, Symbols
from jgutils import functions as jf

if TYPE_CHECKING:
    from jambot.exchanges.exchange import SwaggerExchange
    from jambot.tradesys.broker import Broker
    from jambot.tradesys.trade import Trade
    from jambot.tradesys.wallet import Wallet

log = getlog(__name__)


class BaseOrder(object, metaclass=ABCMeta):
    """Base class to be inherited by backtest or live exchange orders"""
    ts_format = '%Y-%m-%d %H:%M'

    def __init__(
            self,
            qty: Num,
            symbol: Symbol = SYMBOL,
            name: str = '',
            price: float = None,
            offset: float = None,
            order_id: str = None,
            **kw):

        self.status = OrderStatus.PENDING
        self.price_original = price
        self.timestamp_filled = None
        self.qty = qty
        self.name = name
        self.price = price
        self.offset = offset
        self.symbol = symbol
        self.order_id = order_id
        self.fee = 0

    @property
    def qty(self) -> Num:
        return self._qty

    @qty.setter
    def qty(self, val: Num):
        """Set qty and side based on qty"""
        self._qty = float(val)  # TODO confirm this is always rounded from wallet to lot_size
        self._side = TradeSide(np.sign(val))

    def increase_qty(self, qty: Num):
        """Increase absolute order quantity
        - Used for testing only
        """
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
    def is_partially_filled(self):
        return self.status == OrderStatus.PARTIALLYFILLED

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
    def is_limit_open(self) -> bool:
        """If order is a limit_open"""
        return self.is_increase and self.is_limit

    @property
    def is_limit_close(self) -> bool:
        """If order is a limit_close"""
        return self.is_reduce and self.is_limit

    @property
    def is_buy(self) -> bool:
        """If order is buy"""
        return self.side == TradeSide.LONG

    @property
    def is_sell(self) -> bool:
        """If order is sell"""
        return self.side == TradeSide.SHORT

    @property
    def is_manual(self) -> bool:
        """If order is from a manually placed order on Bitmex"""
        return False

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

        _price = f'{self.price:_.{self.symbol.prec}f}' if not self.price is None else None

        return dict(
            ts=self.format_ts(self.timestamp_start),
            # ts_filled=self.format_ts(self.timestamp_filled),
            symbol=self.symbol,
            order_type=str(self.order_type),
            qty=f'{self.qty:+,.{self.symbol.prec_qty}f}',
            price=_price,
            status=self.status,
            name=self.name)

    def as_exch_order(self, exch_name: str = None) -> 'ExchOrder':
        """Convert to ExchOrder"""
        return ExchOrder.from_base_order(order=self, exch_name=exch_name)

    def to_market(self, **kw) -> None:
        """Convert self to market order
        - allow setting any other self params (eg name) with kw
        """
        self.price = None
        self.order_type = OrderType('market')

        jf.set_self(include=kw)

    @property
    def short_stats(self) -> str:
        """Return compressed version of info for messages
        - "XBTUSD | limit_open | $48,852 | +394,800" """
        qty = self.qty if not self.qty is None else 0
        action = '🟢' if qty >= 0 else '🔴'
        price = f'${self.price:,.0f}' if not self.price is None else ''
        return f'{action} {self.symbol} | {self.name} | {qty:+,} | {price}'

    def add(self, lst: Union['Trade', list]) -> 'BaseOrder':
        """Convenience func to add self to trade or list of orders

        Parameters
        ----------
        lst : Union['Trade', list]

        Returns
        -------
        Order
            self
        """
        if not lst is None:
            if isinstance(lst, list):
                lst.append(self)
            else:
                # actually a trade
                lst.add_order(self)

        return self


class ExchOrder(BaseOrder, DictRepr, Serializable):
    """Class to represent bitmex live-trading orders"""
    order_type = ''
    ts_format = '%Y-%m-%d %H:%M:%S'

    # dict to convert exchange order keys
    # TODO should move this to exchange and convert all keys on intake
    # so ExchangeOrder doesn't have to know which exchange its for
    _m_conv = dict(
        bitmex=dict(
            order_type='ordType',
            status='ordStatus',
            order_id='orderID',
            qty='orderQty',
            cum_qty='cumQty',
            price='price',
            stop_px='stopPx',
            symbol='symbol',
            key='clOrdID',
            order_link_id='clOrdID',
            exec_inst='execInst',
            timestamp_start='transactTime',
            avg_price='avgPx',
            name='name',
            side='side'),
        bybit=dict(
            order_type='order_type',
            status='order_status',
            order_id='order_id',
            qty='qty',
            cum_qty='cum_exec_qty',
            price='price',
            stop_px='stop_px',
            symbol='symbol',
            key='order_link_id',
            order_link_id='order_link_id',
            exec_inst='',
            timestamp_start='created_at',
            avg_price='price',
            name='name',
            side='side'))

    def __init__(
            self,
            symbol: Symbol,
            order_type: str,
            status: str = None,
            order_spec_raw: dict = None,
            order_id: str = None,
            timestamp: dt = None,
            timestamp_start: dt = None,
            key: str = None,
            stop_px: float = None,
            name: str = '',
            prevent_market_fill: bool = False,
            exch_name: str = None,
            **kw):

        # convert stop_px to price for stop orders
        if not stop_px is None:
            kw['price'] = stop_px

        max_name_len = 12
        if len(name) > max_name_len:
            raise ValueError(f'Order name too long: {len(name)}, {name}. Max: {max_name_len}')

        super().__init__(symbol=symbol, name=name, **kw)

        order_type = OrderType(order_type)
        status = OrderStatus(status or 'new')

        jf.set_self(exclude=('order_spec',))

    @classmethod
    def from_dict(cls, order_spec: Union[dict, Order], syms: Symbols, exch_name: str = None) -> 'ExchOrder':
        """Create order from exchange order spec dict or BaseOrder"""

        # convenience to convert BaseOrder from strategy
        if isinstance(order_spec, Order):
            return cls.from_base_order(order=order_spec, exch_name=exch_name)

        # try getting exchange from order
        exch_name = exch_name or order_spec.get('exchange', None)

        # convert exchange keys to nice keys, or just use nice keys
        if not exch_name is None:
            _m_conv = cls._m_conv[exch_name]
            # NOTE this could be sketch, m_conv only set if getting data from exchange
            cls.m_conv = _m_conv
            m = {k: order_spec.get(_m_conv[k], order_spec.get(k, None)) for k in _m_conv.keys()}
        else:
            m = copy.copy(order_spec)

        # convert str symbol to Symbol
        m['symbol'] = syms.symbol(m['symbol'], exch_name=exch_name)  # type: ignore

        return cls(**m, exch_name=exch_name, order_spec_raw=order_spec)

    @classmethod
    def from_base_order(cls, order: 'Order', exch_name: str = None) -> 'ExchOrder':
        """Create ExchOrder from base order
        - used to get final/expected orders from strategy and submit/amend
        """

        return cls(
            order_type=order.order_type,
            price=order.price,
            symbol=order.symbol,
            qty=order.qty,
            name=order.name,
            offset=order.offset,
            exch_name=exch_name)

    @classmethod
    def example(cls):
        return cls(symbol=SYMBOL, order_type='limit', qty=-1000, price=50000, name='example_order')

    @classmethod
    def market(cls, **kw) -> 'ExchOrder':
        """Convenience to create market order"""
        return cls(order_type='market', **kw)

    @classmethod
    def limit(cls, **kw) -> 'ExchOrder':
        """Convenience to create market order"""
        return cls(order_type='limit', **kw)

    @classmethod
    def stop(cls, **kw) -> 'ExchOrder':
        """Convenience to create market order"""
        return cls(order_type='stop', **kw)

    @property
    def side_str(self) -> str:
        """Get side name for submitting ByBit orders"""
        return {1: 'Buy', -1: 'Sell'}[self.side]

    @property
    def qty(self) -> Num:
        """Ensure ExchOrders always in multiples of 100 (Bitmex only)
        - Should be able to use lot_size properly now
        - Bybit reduce orders need to use exact qty (no 100 bin_size limit)
        """
        # if self.exch_name == 'bybit' and self.is_reduce:
        #     return self._qty
        # else:
        return f.round_down(n=abs(self._qty), nearest=self.symbol.lot_size) * self.side

    @qty.setter
    def qty(self, val: Num):
        """Set qty and side based on qty
        - NOTE not dry, but can't redefine getter without setter in same class
        """
        self._qty = val
        self._side = TradeSide(np.sign(val))

    @property
    def exec_inst(self) -> list:
        """Create order type specific exec instructions
        - TODO will probably need to allow adding extra exec_inst specs
        """
        lst = []
        if self.is_limit and self.prevent_market_fill:
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
        """Create order spec dict to submit to exchange
        """
        m = dict(
            order_id=self.order_id,
            symbol=self.symbol.upper(),
            order_type=str(self.order_type).title(),
            qty=self.qty,
            key=self.key,
            exec_inst=self.exec_inst_str,
            side=self.side_str,
            time_in_force='GoodTillCancel',
            p_r_price=self.price,
            p_r_qty=self.qty)

        # market order doesn't have price
        # stop needs stopPx only
        if self.is_limit:
            m['price'] = self.price
        elif self.is_stop:
            m['stop_px'] = self.price

        if self.is_reduce:
            m['reduce_only'] = True
            m['close_on_trigger'] = True
        else:
            m['reduce_only'] = False
            m['close_on_trigger'] = False

        return m

    @property
    def is_manual(self) -> bool:
        """If order is from a manually placed order on Bitmex"""
        return self.raw_spec('manual')

    def amend_from_order(self, order: ExchOrder) -> None:
        """Update self.price and self.qty from different order

        Parameters
        ----------
        order : ExchOrder
            order to copy data from
        """
        self.price = order.price
        self.qty = order.qty

    def raw_spec(self, key: str, default: Any = None) -> Any:
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
                try:
                    return self.order_spec_raw[self.m_conv.get(key)]
                except KeyError:
                    if not default is None:
                        return default
                    else:
                        log.warning(f'key "{key}" doesn\'t exist in order_spec_raw')
                        return None
        else:
            raise AttributeError('order_spec_raw not set.')

    def summary_msg(self, exch: 'SwaggerExchange' = None, nearest: float = 0.5) -> str:
        """Get buy/sell price qty summary for discord msg
        -eg "XBT | Sell -2,000 at $44,975.5 (44972.0) | limit_open"
        - TODO this will need precision handled for other symbols

        Returns
        -------
        str
        """
        m = self.order_spec_raw
        if not exch is None:
            avgpx = f.round_down(n=m.get(self._m_conv[exch.exch_name]['avg_price']), nearest=nearest)
        else:
            avgpx = 0

        ordprice = f' (${self.price:,})' if not self.price == avgpx else ''

        stats = ''
        if not exch is None:
            # NOTE this is a bit messy, would prefer properties but dont wanna make em for everything
            if not exch.balance_set:
                exch.set_total_balance()

            stats = f' | Bal: {exch.total_balance_wallet:.4f} | ' \
                + f'PnL: {exch.prev_pnl:.4f}' if self.is_stop or self.is_reduce else ''

        qty = self.qty if not self.is_partially_filled else self.raw_spec('cum_qty')

        action = '🟢' if qty >= 0 else '🔴'

        return '{} {} | {:<4} {:>+8,} at ${:,}{:>12} | {:<12}{}'.format(
            action,
            self.sym_short,
            m['side_str'],
            qty,
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
        """Add exchange + key for ExchOrders"""
        return super().to_dict() | dict(key=self.key)


class Order(BaseOrder, Observer, metaclass=ABCMeta):
    """Order base to simulate backtests"""

    def __init__(
            self,
            order_id: str = None,
            timeout: float = float('inf'),
            trail_close: float = None,
            **kw):

        super().__init__(**kw)
        Observer.__init__(self)

        # these MUST be instance attrs
        self.filled = SignalEvent(float)
        self.cancelled = SignalEvent()
        self.amended = SignalEvent()
        self.timedout = SignalEvent(object)

        # give order unique id
        if order_id is None:
            order_id = str(uuid.uuid1())
        self.order_id = order_id

        self.timeout = timeout
        self.trail_close = trail_close

    @property
    def broker(self) -> 'Broker':
        """Global Broker obj"""
        return self.parent.broker

    @property
    def wallet(self) -> 'Wallet':
        """Global Wallet obj"""
        return self.parent.wallet

    @property
    def is_expired(self) -> bool:
        """Check if order has reached its timeout duration"""
        return self.duration >= self.timeout

    def step(self):
        pass
        # if self.is_expired:
        #     print('TIMED OUT')
        #     self.timedout.emit(self)

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

    def max_qty(self, price: float = None) -> float:
        """get max available qty at current price"""
        price = price or self.price
        return self.wallet.available_quantity(price=price) * self.side  # type: ignore

    def adjust_max_qty(self) -> None:
        """Set qty to max available
        """
        self.broker.amend_order(order=self, qty=self.max_qty())

    def adjust_price(self, price: float) -> None:
        """Adjust price and max qty

        Parameters
        ----------
        price : float
        """

        # can't change close order's qty
        if self.is_reduce:
            qty = None
        else:
            price += -2 * self.symbol.tick_size * self.side  # offset limit_close by eg $1 to keep on inside
            qty = self.max_qty(price)

        self.broker.amend_order(order=self, qty=qty, price=price)

    def to_dict(self) -> dict:
        """Add t_num for strat Orders"""
        t_num = self.parent.trade_num if not self.parent is None else None
        return super().to_dict() | dict(
            ts_filled=self.format_ts(self.timestamp_filled),
            t_num=t_num)


class LimitOrder(Order):
    """Limit order which can be set at price and wait to be filled"""
    order_type = OrderType.LIMIT

    @classmethod
    def example(cls):
        return cls(qty=-1000, price=8888)

    def step(self):

        # adjust price/qty to offset from current close
        if not self.trail_close is None and not self.is_filled:
            price = f.get_price(
                pnl=self.trail_close,
                price=self.c.close,
                side=self.side,
                tick_size=self.symbol.tick_size)
            self.adjust_price(price=price)


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
        """Create StopOrder from LimitOrder

        Parameters
        ----------
        order : Order
        stop_pct : float
            pct to offset from LimitOrder

        Returns
        -------
        StopOrder
        """
        stop_price = f.get_price(
            pnl=abs(stop_pct) * -1,
            price=order.price,
            side=order.side,
            tick_size=order.symbol.tick_size)

        stop_order = StopOrder(
            symbol=order.symbol,
            qty=order.qty * -1,
            price=stop_price,
            name='stop_close')

        return stop_order


def make_order(order_type: Union['OrderType', str], **kw) -> Order:
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
        stop=StopOrder)[str(order_type)]

    return cls(**kw)


def make_orders(
        order_specs: Union[List[dict], dict],
        as_exch_order: bool = False, **kw) -> Union[List[Order], List[ExchOrder]]:
    """Make multiple orders
    - NOTE symbol must be Symbol

    Parameters
    ----------
    order_specs : Union[List[dict], dict]
        list of order_specs dicts
    as_exch_order : bool
        convert to bitmex orders or not

    Returns
    -------
    List[Order] | List[ExchOrder]
        list of initialized Order | ExchOrder objects
    """
    order_specs = jf.as_list(order_specs)

    if not as_exch_order:
        orders = [make_order(**order_spec, **kw) for order_spec in order_specs]

    else:
        # dont want to mix up strat Orders with ExchOrders (strat creates its own order_id)
        orders = [ExchOrder(**order_spec, **kw) for order_spec in order_specs]

    return orders


def make_exch_orders(order_specs: Union[List[dict], dict], exch_name: str, syms: Symbols = None) -> List[ExchOrder]:
    """Create multiple ExchOrders orders from raw bitmex order spec dicts
    - if already dict just return

    Parameters
    ----------
    order_specs : list
        list of dicts, MUST be response from bitmex
    exch_name : str
        force specific exchange (needed for bybit qty)

    Returns
    -------
    List[ExchOrder]
    """

    # used to convert symbol str to Symbol
    if syms is None:
        syms = Symbols()

    return [
        ExchOrder.from_dict(spec, exch_name=exch_name, syms=syms) if not isinstance(spec, ExchOrder) else spec
        for spec in jf.as_list(order_specs)]


def list_to_dict(
        orders: List[Union[Order, ExchOrder]],
        use_ts: bool = False) -> Dict[str, Union[Order, ExchOrder]]:
    """Convenience func to convert list of orders to dict for convenient matching

    Parameters
    ----------
    orders : List[Order | ExchOrder]
        list of Orders or ExchOrders
    key_base : bool, optional default True
        use key_base or full key with timestamp as key

    Returns
    -------
    Dict[str, Order | ExchOrder]
        dict of {order.key_base: order}
    """
    key = 'key' if use_ts else 'key_base'
    return {getattr(o, key): o for o in orders}
