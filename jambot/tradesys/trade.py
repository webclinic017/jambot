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
        # strat: 'Strategy' = None,
        # side: int,
        # target_price: float,
        # contracts: int,
            **kw):

        super().__init__(**kw)

        opened = SignalEvent()
        closed = SignalEvent()

        orders = []
        status = TradeStatus.PENDING
        _side = None
        # active = True
        # filledcontracts = 0
        # contracts = 0
        # entryprice = 0
        # exitprice = 0
        # pnlfinal = 0
        # maxpnl = 0
        # iType = 1
        # sym = strat.sym
        # exitbalance = 0
        # exitcontracts = 0
        # partial = False
        # timedout = False
        # trend_changed = False
        # stopped = False

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

        if self.is_pending:
            self.side = np.sign(qty)
            self.status = TradeStatus.OPEN
        elif self.is_open:
            if self.wallet.qty == 0:
                self.close()

    def market_close(self):
        """Create order to market close all contracts"""
        qty = self.wallet.qty * -1

        if qty == 0:
            return

        order = MarketOrder(
            qty=qty,
            name='market_close',
            symbol=self.symbol)

        self.add_order(order)

    def init(self):
        # entrytarget = price
        # entryprice = entryprice
        # targetcontracts = int(targetcontracts)
        # self.strat = strat
        # sym = strat.sym
        # conf = round(conf, 3)
        # tradenum = strat.tradecount()
        # timeout = strat.timeout
        # self.cdl = self.sym.cdl
        # entrybalance = sym.account.get_balance()

        # if side is None:
        #     status = strat.status
        #     side = strat.get_side()  # TODO: sketch
        # else:
        #     status = side
        #     side = side

        # if not temp:
        #     strat.trades.append(self)

        f.set_self(vars())
        # self.enter()

    def close(self):
        self.status = TradeStatus.CLOSED
        # self.closed.emit()
        self.detach()

    def exit_trade(self):
        # if not exitprice is None:
        #     self.exitprice = exitprice

        self.strat.status = 0
        self.pnlfinal = f.get_pnl(self.side, self.entryprice, self.exitprice)
        self.exitbalance = self.sym.account.get_balance()
        self.i_exit = self.strat.i
        self.active = False

    def close_order(self, price, contracts):

        if contracts == 0:
            return

        self.exitprice = (self.exitprice * self.exitcontracts + price * contracts) / (self.exitcontracts + contracts)

        if self.entryprice == 0:
            raise ValueError('entry price cant be 0!')

        self.sym.account.modify(
            xbt=f.get_pnl_xbt(contracts * -1, self.entryprice, price, self.sym.altstatus),
            timestamp=self.cdl.Index)

        self.exitcontracts += contracts
        self.contracts += contracts

    def close_position(self):
        closeprice = self.sym.cdl.Open
        self.close_order(price=closeprice, contracts=self.contracts * -1)
        self.deactivate_orders(closeall=True)  # this is only Trade_Chop

    # def get_candle(self, i):
    #     return self.candles[i - 1]

    # def add_candle(self, cdl):
    #     self.candles.append(cdl)
    #     self.cdl = cdl

    # def duration(self):
    #     offset = -1 if self.partial else 0
    #     return len(self.candles) + offset

    # def check_timeout(self):
    #     """Check if trade timed out"""
    #     if self.duration() >= self.timeout:
    #         self.timedout = True
    #         self.active = False

    def pnl_acct(self):
        if self.exitbalance == 0:
            return 0
        else:
            return ((self.exitbalance - self.entrybalance) / self.entrybalance)

    def pnl_xbt(self):
        # not used
        return f.get_pnl_xbt(contracts=self.filledcontracts,
                             entryprice=self.entryprice,
                             exitprice=self.exitprice,
                             isaltcoin=self.sym.altstatus)

    def pnl_current(self, c=None):
        if c is None:
            c = self.get_candle(self.duration())
        return f.get_pnl(self.side, self.entryprice, c.Close)

    def pnl_maxmin(self, maxmin, firstonly=False):
        return f.get_pnl(self.side, self.entryprice, self.extremum(self.side * maxmin, firstonly))

    def is_good(self):
        ans = True if self.pnlfinal > 0 else False
        return ans

    def is_stopped(self):
        # NOTE this might be old/not used now that individual orders are used
        ans = True if self.pnl_maxmin(-1) < self.strat.stoppercent else False
        return ans

    def exit_date(self):
        return self.candles[self.duration()].timestamp

    def rescale_orders(self, balance):
        # need to fix 'orders' for trade_chop
        for order in self.orders:
            order.rescale_contracts(balance=balance, conf=self.conf)

    def extremum(self, highlow, firstonly=False):

        # entry candle
        c = self.candles[0]
        with f.Switch(self.status * highlow) as case:
            if case(1, -2):
                if highlow == 1:
                    ext = c.High
                elif highlow == -1:
                    ext = c.Low
            elif case(-1, 2):
                ext = self.entryprice

        if firstonly:
            return ext

        # middle candles
        for i in range(1, self.duration() - 2):
            c = self.candles[i]
            if highlow == 1:
                if c.High > ext:
                    ext = c.High
            elif highlow == -1:
                if c.Low < ext:
                    ext = c.Low

        # exit candle
        c = self.candles[self.duration() - 1]
        with f.Switch(self.status * highlow) as case:
            if case(-1, 2):
                fExt = self.exitprice
            elif case(1, -2):
                if highlow == 1:
                    fExt = c.High
                elif highlow == -1:
                    fExt = c.Low

        ext = fExt if (fExt - ext) * highlow > 0 else ext

        return ext

    def all_orders(self):
        return self.orders

    def exit_order(self):
        return list(filter(lambda x: 'close' in x.name, self.orders))[0]

    def df(self):
        return self.sym.df.iloc[self.i_enter:self.i_exit]

    def print_orders(self, orders=None):
        if orders is None:
            orders = self.all_orders()

        data = []
        cols = ['IDX', 'Type', 'Name', 'Side', 'Price', 'PxOriginal',
                'Cont', 'Active', 'Cancelled', 'Filled', 'Filltype']

        for o in orders:
            ordtype_bot = o.ordarray.letter() if not o.ordarray is None else o.ordtype_bot

            data.append([
                o.index,
                ordtype_bot,
                o.name,
                o.side,
                o.price,
                o.pxoriginal,
                o.contracts,
                o.active,
                o.cancelled,
                o.filled,
                o.ordtype_str()])

        df = pd.DataFrame(data=data, columns=cols)
        # display(df)
        return df
