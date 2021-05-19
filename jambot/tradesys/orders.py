import uuid
from abc import ABCMeta

from .__init__ import *
from .base import Observer, SignalEvent
from .enums import OrderStatus, OrderType, TradeSide


class Order(Observer, metaclass=ABCMeta):

    def __init__(
        self,
        qty: int,
        # ordtype: str,
        price: float,
        symbol: str = None,
        # side: int = None,
        # ordtype_bot: int = None,
        reduce_only: bool = False,
        # trade: 'TradeBase' = None,
        # sym: 'Backtest' = None,
        order_id: str = None,
        # activate: bool = False,
        name: str = '',
        exec_inst: list = None,
        is_live: bool = False,
        timeout: int = float('inf')
    ):

        super().__init__()

        filled = SignalEvent(int)
        cancelled = SignalEvent()
        ammended = SignalEvent(float)

        # give order unique id
        if order_id is None:
            order_id = str(uuid.uuid4())

        status = OrderStatus.PENDING
        name = name.lower()
        # orderID = ''
        # delaytime = None
        filled_time = None
        # marketfilled = False
        # cancelled = False
        # livedata = []

        if qty == 0:
            raise ValueError('Order quantity cannot be 0.')

        # is_stop = True if ordtype == 'Stop' else False

        # if not trade is None:
        #     manual = False
        #     sym = trade.strat.sym
        #     slippage = trade.strat.slippage
        #     trade.orders.append(self)
        # else:
        #     manual = True
        #     if name == '':
        #         name = 'manual'

        # live trading
        # if not sym is None:
        #     decimal_figs = sym.decimal_figs
        #     symbolbitmex = sym.symbolbitmex
        # else:
        #     decimal_figs = 0
        #     symbolbitmex = symbol
        #     if symbol is None:
        #         raise NameError('Symbol required!')

        # decimaldouble = float(f'1e-{decimal_figs}')

        f.set_self(vars())
        # self.price = self.final_price(price)
        self.price_original = self.price
        # self.set_key()

        if is_live:
            self.set_live_data()

    @property
    def side(self):
        """Return qty positive or negative"""
        return TradeSide(np.sign(self.qty))

    @property
    def is_filled(self):
        return self.status == OrderStatus.FILLED

    @property
    def is_expired(self):
        """Check if order has reached its timeout duration"""
        return self.duration >= self.timeout

    def step(self):
        """Check if execution price hit and fill"""
        pass

    def fill(self):
        """Decide if adding or subtracting qty"""
        self.filled_time = self.timestamp
        self.filled.emit(self.qty)  # NOTE not sure if qty needed
        self.status = OrderStatus.FILLED
        self.detach()

        # market filling
        # if not price is None:
        #     self.price = price
        #     self.marketfilled = True

        # price = self.check_stop_price()

        # self.open_(price=price) if self.add_subtract == 1 else self.close(price=price)

    def cancel(self):
        """Cancel order"""
        self.cancelled.emit()

    # def final_price(self, price=None):
    #     # NOTE not really sure why this is necessary
    #     if self.ordtype == 'Market':
    #         return None
    #     elif self.manual:
    #         return price
    #     else:
    #         if price is None:
    #             price = self.price
    #         return round(round(price, self.decimal_figs) + self.decimaldouble * self.side * -1, self.decimal_figs)

    def ordtype_str(self):
        # v sketch
        if self.filled:
            ans = 'L' if not self.marketfilled else 'M'
        else:
            ans = pd.NA

        return ans

    def set_name(self, name):
        self.name = name
        self.set_key()

    def set_key(self):
        side = self.side if self.trade is None else self.trade.side

        self.key = f.key(self.symbolbitmex, self.name, side, self.ordtype)
        self.clOrdID = '{}-{}'.format(self.key, int(time()))

    def set_live_data(self, exec_inst: list = None):
        """Add exec inst to order for live trading
        - TODO haven't tested this at all, needs to be fixed for live trading
        """
        if exec_inst is None:
            exec_inst = []

        if not isinstance(exec_inst, list):
            exec_inst = [exec_inst]

        if self.ordtype == 'Limit':
            self.exec_inst.append('ParticipateDoNotInitiate')
        if 'stop' in self.name:
            self.exec_inst.append('IndexPrice')
        if 'close' in self.name:
            self.exec_inst.append('Close')

    def stoppx(self):
        return self.price * (1 + self.slippage * self.side)

    def set_price(self, price):
        if not self.filled:
            self.price = price

    def check_stop_price(self):
        """Use price if Limit/Market, else use slippage price"""
        price = self.price if not self.is_stop else self.stoppx()
        return price

    def open_(self, price):
        """Increase trade qty and adjust trade entryprice
        NOTE some of this should be handled by trade object
        """
        t = self.trade
        qty = self.qty

        if qty == 0:
            return

        t.entryprice = (t.entryprice * t.qty + price * qty) / (t.qty + qty)
        t.qty += qty

    def close(self, price: float):
        """Close order and reduce total trade qty"""
        self.trade.close_order(price=price, qty=self.qty)

    def is_active(self, c):
        if not self.delaytime is None:
            active = True if self.active and c.Index >= self.delaytime else False
        else:
            active = self.active

        return active

    def activate(self, c, delay: int = 0):
        """Make order active after "delay" hours"""
        self.delaytime = c.Index + delta(hours=delay)
        self.active = True

    # def cancel(self):
    #     self.active = False
    #     self.cancelled = True
    #     self.filledtime = self.sym.cdl.Index
    #     if not self.ordarray is None:
    #         self.ordarray.openorders -= 1

    def rescale_contracts(self, balance, conf=1):
        self.qty = int(conf * f.get_contracts(
            xbt=balance,
            leverage=self.trade.strat.lev,
            entryprice=self.price,
            side=self.side,
            isaltcoin=self.sym.altstatus))

    def intake_live_data(self, livedata):
        self.livedata = livedata
        self.orderID = livedata['orderID']

    def append_execinst(self, m):
        if self.exec_inst:
            if isinstance(self.exec_inst, list):
                m['execInst'] = ','.join(self.exec_inst)
            else:
                m['execInst'] = self.exec_inst

        return m

    def amend_order(self):
        m = {}
        m['orderID'] = self.orderID
        m['symbol'] = self.sym.symbolbitmex
        m['orderQty'] = self.qty
        m = self.append_execinst(m)

        with f.Switch(self.ordtype) as case:
            if case('Limit'):
                m['price'] = self.price
            elif case('Stop'):
                m['stopPx'] = self.price

        return m

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

    # def to_dict(self) -> dict:
    #     m = self.new_order()
    #     m['qty'] = self.qty
    #     m['side'] = self.side
    #     m['name'] = self.name
    #     return m
    def to_dict(self) -> dict:

        return dict(
            # order_id=self.order_id,
            symbol=self.symbol,
            order_type=str(self.order_type),
            qty=self.qty,
            price=self.price,
            status=self.status,
            name=self.name
        )

    def new_order(self) -> dict:
        m = {}
        m['symbol'] = self.symbolbitmex
        m['orderQty'] = self.qty
        m['clOrdID'] = self.clOrdID
        m['ordType'] = self.ordtype
        m = self.append_execinst(m)

        with f.Switch(self.ordtype) as case:
            if case('Limit'):
                m['price'] = self.price
            elif case('Stop'):
                m['stopPx'] = self.price
            elif case('Market'):
                m['ordType'] = self.ordtype

        return m

    def __str__(self) -> str:
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self) -> str:
        return str(self)


class LimitOrder(Order):
    """Limit order which can be set at price and wait to be filled"""
    trigger_switch = 1
    order_type = OrderType.LIMIT

    # def __init__(self, **kw):
    #     super().__init__(**kw)

    @classmethod
    def example(cls):
        return cls(qty=1000, price=8888)


class MarketOrder(Order):
    """Market order which will be executed immediately"""
    trigger_switch = None
    order_type = OrderType.MARKET
    # price = None

    def __init__(self, price: float = None, **kw):
        super().__init__(price=price, **kw)


class StopOrder(Order):
    """Stop order which can be used to ender or exit positions"""
    trigger_switch = -1
    order_type = OrderType.STOP

    # def __init__(self, **kw):
    #     super().__init__(**kw)


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


def make_orders(order_specs: list, **kw) -> list:
    """Make multiple orders

    Parameters
    ----------
    order_specs : list
        list of order_specs dicts

    Returns
    -------
    list
        list of initialized Order objects
    """
    return [make_order(**order_spec, **kw) for order_spec in order_specs]
