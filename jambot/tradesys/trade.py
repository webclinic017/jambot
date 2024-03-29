import operator as opr
from typing import *

import numpy as np
import pandas as pd

from jambot import Num, display
from jambot import functions as f
from jambot import getlog
from jambot.tradesys.base import Observer, SignalEvent
from jambot.tradesys.enums import TradeSide, TradeStatus
from jambot.tradesys.exceptions import InvalidTradeOperationError
from jambot.tradesys.orders import MarketOrder, Order
from jambot.tradesys.symbols import Symbol

if TYPE_CHECKING:
    from jambot.tradesys.broker import Broker

log = getlog(__name__)


class Trade(Observer):
    """Trade represents a sequence of orders
    - Must begin and end with 0 quantity
    """

    def __init__(self, symbol: Symbol, broker: 'Broker', **kw):
        super().__init__(**kw)
        self.opened = SignalEvent()
        self.closed = SignalEvent()

        self.orders = []  # type: List[Order]
        self.status = TradeStatus.PENDING
        self._side = TradeSide.NEUTRAL
        self.qty_filled = 0
        self.wallet = broker.get_wallet(symbol=symbol)
        self.entry_balance = None
        self.exit_balance = None
        self.symbol = symbol
        self.broker = broker
        self.trade_num = -1  # not init

    @property
    def side(self) -> int:
        return self._side

    @side.setter
    def side(self, val: int):
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
    def pnl(self) -> float:
        """PnL % of trade"""
        return f.get_pnl(self.side, self.entry_price, self.exit_price)

    @property
    def pnl_acct(self) -> float:
        """Balance % change of account"""
        if self.exit_balance and self.entry_balance:
            return (self.exit_balance - self.entry_balance) / self.entry_balance
        else:
            return 0

    @property
    def is_good(self) -> bool:
        return self.pnl > 0

    def _filter_orders(self, _type: str = 'entry') -> List[Order]:
        """Filter filled entry or exit orders

        Parameters
        ----------
        _type : str, optional
            entry|exit, default 'entry'

        Returns
        -------
        List[Order]
            list of filtered orders
        """
        op = dict(
            entry=opr.eq,
            exit=opr.ne)[_type]

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

    @property
    def fees(self):
        """Return summed order fees"""
        return sum([o.fee for o in self.orders])

    def add_orders(self, orders: list):
        """Add multiple orders"""
        for order in orders:
            self.add_order(order)

    def add_order(self, order: 'Order') -> None:
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

    def on_fill(self, qty: Num, *args) -> None:
        """Perform action when any orders filled"""
        if self.is_closed:
            log.warning(f'trade already closed: {self.trade_num}')
            print('qty:', qty, 'status:', self.status, 'wallet qty:', self.wallet.qty)

        self.qty_filled += qty

        if self.is_pending:
            self.side = np.sign(qty)
            self.status = TradeStatus.OPEN
            self.entry_balance = self.wallet.balance
        elif self.is_open:
            if self.wallet.is_zero:
                self.close()

    def cancel_open_orders(self) -> None:
        """Cancel all open orders"""
        for order in self.open_orders:
            self.broker.cancel_order(order)

    def market_close(self) -> None:
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
                raise InvalidTradeOperationError(
                    qty_open=self.qty_filled,
                    trade_num=self.trade_num,
                    orders=self.orders)

            self.status = TradeStatus.CLOSED
            self.exit_balance = self.wallet.balance
            self.detach_from_parent()
            self.closed.emit()

    def pnl_maxmin(self, maxmin, firstonly=False):
        return f.get_pnl(self.side, self.entry_price, self.extremum(self.side * maxmin, firstonly))

    # def rescale_orders(self, balance):
    #     # need to fix 'orders' for trade_chop
    #     for order in self.orders:
    #         order.rescale_contracts(balance=balance, conf=self.conf)

    # def df(self) -> pd.DataFrame:
    #     """Show df of candles for trade's duration"""
    #     return self.bm.df.iloc[self.i_enter:self.i_exit]

    def to_dict(self) -> dict:
        return dict(
            side=self.side_planned,
            qty=sum(o.qty for o in self.entry_orders),
            entry_price=f'{self.entry_price:_.{self.symbol.prec}f}',
            exit_price=f'{self.exit_price:_.{self.symbol.prec}f}',
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
            pnl_acct=self.pnl_acct,
            bal=self.exit_balance,
            fees=self.fees,
            status=self.status,
            t_num=self.trade_num)

    @property
    def df_orders(self) -> pd.DataFrame:
        data = [o.to_dict() for o in self.orders]
        return pd.DataFrame.from_dict(data)

    def show_orders(self) -> None:
        display(self.df_orders)
