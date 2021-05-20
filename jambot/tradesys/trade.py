from abc import ABCMeta, abstractmethod

from . import orders as ords
from .__init__ import *
from .base import Observer, SignalEvent
from .broker import Broker
from .enums import TradeSide, TradeStatus
from .orders import MarketOrder, Order
from .wallet import Wallet


class Trade(Observer):
    """Trade represents a sequence of orders
    - Must begin and end with 0 quantity
    """

    def __init__(
            self,
            symbol: str,
            broker: 'Broker',
            **kw):

        super().__init__(**kw)

        opened = SignalEvent()
        closed = SignalEvent()

        orders = []
        status = TradeStatus.PENDING
        _side = TradeSide.NEUTRAL

        wallet = broker.get_wallet(symbol=symbol)

        f.set_self(vars())

    @property
    def side(self):
        return self._side

    @side.setter
    def side(self, val):
        """Set side as TradeSide"""
        self._side = TradeSide(val)

    @property
    def is_pending(self):
        return self.status == TradeStatus.PENDING

    @property
    def is_open(self):
        return self.status == TradeStatus.OPEN

    @property
    def is_closed(self):
        return self.status == TradeStatus.CLOSED

    def step(self):
        pass

    def add_orders(self, orders: list):
        """Add multiple orders"""
        for order in orders:
            self.add_order(order)

    @property
    def num_orders(self):
        return len(self.orders)

    @property
    def pnl(self):
        return f.get_pnl(self.side, self.entry_price, self.exit_price)

    def add_order(self, order: 'Order'):
        """Add order and connect filled method

        Parameters
        ----------
        order : Order
            order to connect
        """
        self.attach(order)
        self.orders.append(order)
        order.filled.connect(self.on_fill)
        self.broker.submit(order)

    # def make_order(self, order_spec: dict):
    #     """Make single order and attach self to filled signal"""

    #     order = ords.make_order(**order_spec)
    #     order.filled.connect(self.on_fill)

    #     return order

    def on_fill(self, qty: int, *args):
        """Perform action when any orders filled"""

        # chose side based on first order filled
        self.entry_price = self.wallet.entry_price
        self.exit_price = self.wallet.exit_price

        if self.is_pending:
            self.side = np.sign(qty)
            self.status = TradeStatus.OPEN
        elif self.is_open:
            if self.wallet.qty == 0:
                self.close()

    def market_close(self):
        """Create order to market close all qty"""
        qty = self.wallet.qty * -1

        if qty == 0:
            return

        order = MarketOrder(
            qty=qty,
            name='market_close',
            symbol=self.symbol)

        self.add_order(order)

    def close(self):
        self.status = TradeStatus.CLOSED
        # self.closed.emit()
        self.detach()

    @property
    def same_orders(self):
        """Return orders which match self trade side"""
        return [o for o in self.orders if o.side == self.side]

    def pnl_maxmin(self, maxmin, firstonly=False):
        return f.get_pnl(self.side, self.entry_price, self.extremum(self.side * maxmin, firstonly))

    def is_good(self):
        return True if self.pnl > 0 else False

    def is_stopped(self):
        # NOTE this might be old/not used now that individual orders are used
        ans = True if self.pnl_maxmin(-1) < self.strat.stoppercent else False
        return ans

    def rescale_orders(self, balance):
        # need to fix 'orders' for trade_chop
        for order in self.orders:
            order.rescale_contracts(balance=balance, conf=self.conf)

    def exit_order(self):
        return list(filter(lambda x: 'close' in x.name, self.orders))[0]

    def df(self):
        return self.bm.df.iloc[self.i_enter:self.i_exit]

    def to_dict(self):
        return dict(
            side=self.side,
            qty=sum(o.qty for o in self.same_orders),
            entry_price=f'{self.entry_price:_.0f}',
            exit_price=f'{self.exit_price:_.0f}',
            pnl=f'{self.pnl:.2%}')
