import operator as opr

from .__init__ import *
from .base import Observer, SignalEvent
from .broker import Broker
from .enums import TradeSide, TradeStatus
from .orders import MarketOrder, Order


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
        exit_balance = None
        qty_filled = 0

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
    def side_planned(self):
        """Get "theoretical" side of trade if all entry orders had filled"""
        # NOTE kinda sketchy, just assume first order added was the side we wanted for now
        return self.orders[0].side

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

    @property
    def num_orders(self):
        return len(self.orders)

    @property
    def pnl(self):
        return f.get_pnl(self.side, self.entry_price, self.exit_price)

    @property
    def is_good(self):
        return self.pnl > 0

    def _filter_orders(self, _type: str = 'entry'):
        """Filter filled entry or exit orders

        Parameters
        ----------
        _type : str, optional
            entry|exit, default 'entry'

        Returns
        -------
        list
            list of filtered orders
        """
        op = dict(
            entry=opr.eq,
            exit=opr.ne).get(_type)

        return [o for o in self.orders if op(o.side, self.side) and o.is_filled]

    @property
    def entry_orders(self):
        """Return orders which match self trade side"""
        return self._filter_orders('entry')

    @property
    def exit_orders(self):
        """Return orders which do not match self trade side"""
        return self._filter_orders('exit')

    @property
    def open_orders(self):
        return [o for o in self.orders if o.is_open]

    def _weighted_price(self, _type: str = 'entry') -> float:
        """Calc weighted price for entry/exit orders

        Parameters
        ----------
        _type : str, optional
            entry|exit, default 'entry'

        Returns
        -------
        float
            price weighted by quantity
        """
        orders = self._filter_orders(_type)
        prices = [o.price for o in orders]
        qtys = [o.qty for o in orders]

        if not prices:
            # NOTE not sure if this is best solution yet
            # print(prices)
            # print(qtys)
            return 0

        return np.average(prices, weights=qtys)

    @property
    def entry_price(self):
        return self._weighted_price('entry')

    @property
    def exit_price(self):
        return self._weighted_price('exit')

    @property
    def qty(self):
        """Return quantity of contracts"""
        return sum([o.qty for o in self.entry_orders])

    def add_orders(self, orders: list):
        """Add multiple orders"""
        for order in orders:
            self.add_order(order)

    def add_order(self, order: 'Order'):
        """Add order and connect filled method

        Parameters
        ----------
        order : Order
            order to connect
        """
        if order is None:
            raise ValueError('Order cannot be none!')

        self.attach_listener(order)
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
        # self.entry_price = self.wallet.entry_price
        # self.exit_price = round(self.wallet.exit_price, 1)
        self.qty_filled += qty

        if self.is_pending:
            self.side = np.sign(qty)
            self.status = TradeStatus.OPEN
        elif self.is_open:
            if self.wallet.is_zero:
                self.close()

    def cancel_open_orders(self):
        """Cancel all open orders"""
        for order in self.open_orders:
            self.broker.cancel_order(order)

    def market_close(self):
        """Create order to market close all qty, submit to broker"""

        qty = self.wallet.qty_opp

        if not qty == 0:
            order = MarketOrder(
                qty=qty,
                name='market_close',
                symbol=self.symbol)

            self.add_order(order)
            order.filled.connect(lambda: print('market_close filled'))

        self.close()

    def close(self, *args):
        """close trade"""
        self.cancel_open_orders()

        if not self.is_closed:
            if not self.qty_filled == 0:
                raise ValueError(f'Cant close trade [{self.trade_num}] with [{self.qty_filled}] contracts open!')

            self.status = TradeStatus.CLOSED
            self.exit_balance = self.wallet.balance
            self.detach_from_parent()
            self.closed.emit()

    def pnl_maxmin(self, maxmin, firstonly=False):
        return f.get_pnl(self.side, self.entry_price, self.extremum(self.side * maxmin, firstonly))

    def rescale_orders(self, balance):
        # need to fix 'orders' for trade_chop
        for order in self.orders:
            order.rescale_contracts(balance=balance, conf=self.conf)

    def df(self) -> pd.DataFrame:
        """Show df of candles for trade's duration"""
        return self.bm.df.iloc[self.i_enter:self.i_exit]

    def to_dict(self) -> dict:
        return dict(
            side=self.side_planned,
            qty=sum(o.qty for o in self.entry_orders),
            entry_price=f'{self.entry_price:_.0f}',
            exit_price=f'{self.exit_price:_.0f}',
            pnl=f'{self.pnl:.2%}')

    def dict_stats(self) -> dict:
        """Dict of statistics, useful for creating a df of all trades"""
        return dict(
            ts=self.timestamp_start,
            side=self.side_planned,
            dur=self.duration,
            entry=self.entry_price,
            exit=self.exit_price,
            qty=self.qty,
            pnl=self.pnl,
            bal=self.exit_balance,
            status=self.status,
            t_num=self.trade_num)

    @property
    def df_orders(self) -> pd.DataFrame:
        data = [o.to_dict() for o in self.orders]
        return pd.DataFrame.from_dict(data)

    def show_orders(self) -> None:
        display(self.df_orders)
