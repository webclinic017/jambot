import pandas as pd

from .. import backtest as bt
from .. import functions as f
from .. import signals as sg
from ..backtest import OrdArray, Order

try:
    from IPython.display import display
except ModuleNotFoundError:
    pass


class Strategy(bt.Strategy):
    def __init__(self, speed=(36, 36), weight=1, norm=(1, 4), speedtp=(36, 36)):
        super().__init__(weight=weight)
        self.name = 'chop'
        self.speed = speed
        self.speedtp = speedtp
        self.norm = norm
        # anchordist > 0 - 0.02, step 0.002
        # Order/Stop spread,

    def init(self, bm):
        self.bm = bm

        # unused, these moved into classes
        # bm.df = setTradePrices(self.name, bm.df, speed=self.speed)
        # bm.df = setTradePrices('tp', bm.df, speed=self.speedtp)
        # bm.df = setVolatility(bm.df, norm=self.norm)

    def decide(self, c):
        self.cdl = c

        if abs(self.status) == 1:
            self.trade.check_orders(c)
            if not self.trade.active:
                self.trade.exit_()
                self.trade = None
        else:
            if c.high >= c.chop_high:
                self.status = -1
                self.enter_trade(c.chop_high)
            elif c.low <= c.chop_low:
                self.status = 1
                self.enter_trade(c.chop_low)

    def init_trade(self, entry_price, side, balance=None):
        if balance is None:
            balance = self.bm.account.get_balance()

        qty = f.get_contracts(balance * self.weight, self.lev, entry_price, side, self.bm.altstatus)

        trade = Trade(c=self.cdl)
        trade.init(entry_price, qty, self, side=side)
        return trade

    def enter_trade(self, entry_price):
        self.trade = self.init_trade(entry_price, self.status)
        self.trade.check_orders(self.cdl)

    def get_anchor_price(self, anchorstart, norm, side):
        return anchorstart * (1 + norm * 0.005 * side * -1)

    def get_next_ord_arrays(self, anchorprice, c, side, trade=None):

        orders = OrdArray(ordtype_bot=1,
                          anchorprice=anchorprice,
                          orderspread=0.002 * c.norm,
                          trade=trade,
                          activate=True)

        stops = OrdArray(ordtype_bot=2,
                         anchorprice=f.get_price(
                             -0.01 * c.norm,
                             orders.maxprice,
                             side),
                         orderspread=0.002 * c.norm,
                         trade=trade,
                         activate=False)

        outerprice = c.tp_high if side == 1 else c.tp_low

        takeprofits = OrdArray(ordtype_bot=3,
                               anchorprice=trade.anchorstart,
                               outerprice=outerprice,
                               orderspread=0,
                               trade=trade,
                               activate=False)

        return [orders, stops, takeprofits]

    def final_orders(self, u, weight):
        lstOrders = []
        balance = u.total_balance_wallet * weight
        remainingcontracts = u.get_position(self.bm.symbolbitmex)['currentQty']
        # print(remainingcontracts)

        if not self.trade is None:
            # we should be in a trade
            t = self.trade

            # rescale qty to reflect actual user balance
            targetcontracts = f.get_contracts(balance, self.lev, t.anchorstart, t.side, self.bm.altstatus)

            lstOrders.extend(t.orders.getUnfilledOrders(targetcontracts))
            lstOrders.extend(t.stops.getUnfilledOrders(targetcontracts, remainingcontracts))
            lstOrders.extend(t.takeprofits.getUnfilledOrders(targetcontracts, remainingcontracts))

        else:
            # not in a trade, need upper and lower order/stop arrays
            c = self.cdl

            trade_long = self.init_trade(c.chop_low, 1, balance=balance)
            lstOrders.extend(trade_long.orders.orders)
            lstOrders.extend(trade_long.stops.orders)

            trade_short = self.init_trade(c.chop_high, -1, balance=balance)
            lstOrders.extend(trade_short.orders.orders)
            lstOrders.extend(trade_short.stops.orders)

        return self.checkfinalorders(lstOrders)

    def print_trades(self, maxmin=0, maxlines=-1):
        data = []
        cols = ['N', 'timestamp', 'Sts', 'Dur', 'Anchor', 'Entry', 'Exit', 'Contracts', 'Filled', 'Pnl', 'Balance']
        for i, t in enumerate(self.trades):
            if not maxmin == 0 and maxmin * t.pnlfinal <= 0:
                continue

            data.append([
                t.tradenum,
                '{:%Y-%m-%d %H}'.format(t.candles[0].timestamp),
                t.status,
                t.duration(),
                '{:,.0f}'.format(t.anchorstart),
                '{:,.0f}'.format(t.entry_price),
                '{:,.0f}'.format(t.exit_price),
                '({:,} / {:,})'.format(t.filledcontracts, t.targetcontracts),
                t.all_filled(),
                '{:.2%}'.format(t.pnlfinal),
                round(t.exitbalance, 2)
            ])

            if i == maxlines:
                break

        df = pd.DataFrame(data=data, columns=cols)
        display(df)


class Trade(bt.Trade):
    def __init__(self, c):
        super().__init__()
        self.numorders = 4
        self.cdl = c

    def enter(self):
        self.anchorstart = self.entry_price
        self.entry_price = 0

        c = self.cdl

        self.anchorprice = self.strat.get_anchor_price(self.anchorstart, c.norm, self.status)

        lst = self.strat.get_next_ord_arrays(self.anchorprice, c, self.status, self)
        self.orders = lst[0]
        self.stops = lst[1]
        self.takeprofits = lst[2]

    def exit_(self):
        self.filledcontracts = self.orders.filledcontracts
        self.exit_trade()

    def check_orders(self, c):
        self.add_candle(c)

        if self.duration() == 5:  # 5 is arbitrary
            self.deactivate_orders()
        elif self.duration() == 40:
            self.close_position()

        self.orders.check_orders(c)
        self.stops.check_orders(c)
        self.takeprofits.check_orders(c)

        if not self.orders.active and self.qty == 0:
            self.active = False  # > then exit trade??

        if (not self.stops.active) or (not self.takeprofits.active):
            self.active = False

    def deactivate_orders(self, closeall=False):
        if not closeall:
            for i, order in enumerate(self.orders.orders):
                if not order.filled:
                    order.cancel()
                    self.takeprofits.orders[i].cancel()
                    self.stops.orders[i].cancel()
        else:
            for order in self.all_orders():
                if not order.filled:
                    order.cancel()

    def all_filled(self):
        return '{}-{}-{}'.format(
            self.orders.strfilled(),
            self.stops.strfilled(),
            self.takeprofits.strfilled())

    def all_orders(self):
        lst = []
        lst.extend(self.orders.orders)
        lst.extend(self.stops.orders)
        lst.extend(self.takeprofits.orders)
        return lst

    def print_all_orders(self):
        self.strat.print_orders(self.all_orders())
