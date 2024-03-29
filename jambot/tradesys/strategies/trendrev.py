from jambot import comm as cm
from jambot import functions as f

from ... import signals as sg
from .. import backtest as bt
from ..orders import Order
from ..trade import Trade
from .__init__ import *  # noqa
from .base import StrategyBase


class Strategy(StrategyBase):
    def __init__(self, speed=(8, 8), weight=1, norm=(0.004, 0.024), lev=5):
        super().__init__(weight, lev)
        self.name = 'trendrev'
        self.stop_pct = -0.03
        self.speed = speed
        self.norm = norm
        self.lasthigh, self.lastlow = 0, 0
        self.timeout = 40
        self.slippage = 0

    def init(self, bm):
        self.bm = bm
        df = bm.df
        self.a = self.bm.account

        self.vty = sg.Volatility(weight=1, norm=self.norm)
        df = df.pipe(self.vty.add_signal)

        macd = sg.MACD(weight=1)
        ema = sg.EMA(weight=1)
        emaslope = sg.EMASlope(weight=1, p=50, slope=5)
        df = self.conf.add_signal(df=df, signals=[macd, ema, emaslope],
                                  trendsignals=[ema])

        # NOTE kinda sketch for now, manually adding signals this way to avoid adding to 'conf' signal
        self.trend = sg.Trend(signal_series='ema_trend', speed=self.speed)
        df = df.pipe(self.trend.add_signal)

        self.df = df
        self.bm.df = df

    def exit_trade(self):
        t = self.trade
        c = self.cdl

        if not t.stopped and t.limitopen.filled and not t.limitclose.filled:
            t.stop.cancel()
            t.limitclose.fill(c=c, price=c.close)
            self.unfilledtrades += 1

        t.exit_trade()
        self.trade = None

    def enter_trade(self, side, entry_price):
        self.trade = self.init_trade(trade=TradeRev(), side=side, entry_price=entry_price)
        t = self.trade
        t.check_orders(self.bm.cdl)
        if not t.active:
            self.exit_trade()

    def decide(self, c):
        self.cdl = c
        self.i = c.Index
        pxhigh, pxlow = c.pxhigh, c.pxlow

        # if we exit a trade and limitclose isnt filled, limitopen may wait and fill in next
        # # candles, but current trade gets 0 profit. Need to exit properly.

        # Exit TradeRev
        if not self.trade is None:
            t = self.trade

            if t.side == 1:
                if c.high > max(pxhigh, self.lasthigh):
                    t.active = False
            else:
                if c.low < min(pxlow, self.lastlow):
                    t.active = False

            t.check_orders(c)
            if not t.active:
                self.exit_trade()

            # need to make trade active=False if limitclose is filled?
            # order that candle moves is important
            # TODO: need enter > exit > enter all in same candle

        # Enter TradeRev
        enterhigh, enterlow = self.lasthigh, self.lastlow
        # enterhigh, enterlow = pxhigh, pxlow
        if self.trade is None:
            if c.high > enterhigh:
                self.enter_trade(-1, enterhigh)
            elif c.low < enterlow:
                self.enter_trade(1, enterlow)

        self.lasthigh, self.lastlow = pxhigh, pxlow

    def final_orders(self, u, weight):
        symbol = self.bm.symbolbitmex
        balance = u.balance() * weight
        pos = bt.Position(qty=u.get_position(symbol)['currentQty'])
        pos.add_order(u.get_orders(symbol=symbol, manual_only=True))

        t_prev = self.trades[-2]
        t_current = self.trades[-1]
        t_current.rescale_orders(balance=balance)

        prevclose = t_prev.limitclose
        limitopen = t_current.limitopen
        stopclose = t_current.stop
        limitclose = t_current.limitclose

        # PREVCLOSE - Check if position is still open with correct side/qty
        if (prevclose.marketfilled
            and pos.side() == t_prev.side
                and t_current.duration() <= 4):
            pos.add_order(Order(
                qty=pos.qty * -1,
                ordtype='Market',
                name='marketclose',
                trade=t_prev))

        # OPEN
        if limitopen.filled:
            if (limitopen.marketfilled
                and pos.qty == 0
                    and t_current.duration() == 4):
                pos.add_order(Order(
                    qty=limitopen.qty,
                    ordtype='Market',
                    name='marketopen',
                    trade=t_current))
        else:
            pos.add_order(limitopen)  # Only till max 4 candles into trade

        # STOP
        if not stopclose.filled:
            pos.add_order(stopclose)

        # CLOSE - current
        if not pos.qty == 0:
            if t_current.timedout:
                pos.add_order(Order(
                    qty=limitclose.qty,
                    ordtype='Market',
                    name='marketclose',
                    trade=t_current))
            else:
                pos.add_order(limitclose)

            if stopclose.qty == 0 or not stopclose in pos.orders:
                cm.discord(msg='Error: no stop for current position', channel='err')

        # NEXT - Init next trade to get next limitopen and stop
        c = self.cdl
        t_next = self.init_trade(
            side=t_current.side * -1,
            entry_price=(c.pxhigh if t_current.side == 1 else c.pxlow),
            balance=balance,
            temp=True,
            trade=TradeRev())

        pos.add_order(t_next.limitopen)
        pos.add_order(t_next.stop)

        return pos.final_orders()


class TradeRev(Trade):
    def __init__(self):
        super().__init__()
        self.stopped = False
        self.enteroffset = 0

    def closeprice(self):
        c = self.strat.cdl
        price = c.pxhigh if self.side == 1 else c.pxlow
        # self.enteroffset = self.strat.vty.final(i=c.Index)
        self.enteroffset = c.norm_ema

        # TODO: move this to a 'slippage price' function
        return round(price * (1 + self.enteroffset * self.side), self.bm.decimal_figs)

    def enter(self):
        c = self.strat.cdl
        self.stop_pct = self.strat.stop_pct
        # self.enteroffset = self.strat.vty.final(i=c.Index)
        self.enteroffset = c.norm_ema

        limitbuyprice = self.entrytarget * (1 + self.enteroffset * self.side * -1)
        limitcloseprice = self.closeprice()
        self.stoppx = f.get_price(self.stop_pct, limitbuyprice, self.side)

        qty = int(self.targetcontracts * self.conf)

        self.limitopen = Order(
            price=limitbuyprice,
            side=self.side,
            qty=qty,
            activate=True,
            ordtype_bot=1,
            ordtype='Limit',
            name='limitopen',
            trade=self)

        self.stop = Order(
            price=self.stoppx,
            side=self.side * -1,
            qty=qty,
            activate=False,
            ordtype_bot=2,
            ordtype='Stop',
            name='stop',
            reduce=True,
            trade=self)

        self.limitclose = Order(
            price=limitcloseprice,
            side=self.side * -1,
            qty=qty,
            activate=False,
            ordtype_bot=3,
            ordtype='Limit',
            name='limitclose',
            reduce=True,
            trade=self)

    def check_position_closed(self):
        if self.limitclose.filled:
            self.active = False

    def check_orders(self, c):
        # trade stays active until pxlow is hit, strat controlls
        # filling order sets the trade's actual entry_price
        # filling close or stop order sets trade's exit price

        self.add_candle(c)

        for o in self.orders:
            if o.is_active(c=c) and not o.filled:
                o.check(c)

                # MARKET OPEN
                # Don't market fill if trade would have closed on this candle
                # if order not filled after 4 candles, fill at close price
                if (o.ordtype_bot == 1
                    and self.active
                    and self.duration() == 4
                        and not o.filled):
                    o.fill(c=c, price=c.close)

                if o.filled:
                    if o.ordtype_bot == 1:
                        self.filledcontracts = o.qty
                        delay = 1 if o.marketfilled else 0
                        self.limitclose.activate(c=c, delay=delay)
                        self.stop.activate(c=c, delay=0)
                    elif o.ordtype_bot == 2:
                        self.limitclose.cancel()  # make limitclose filledtime be end of trade
                        self.stopped = True
                        self.pnlfinal = f.get_pnl(self.side, self.entry_price, self.exit_price)
                        self.exitbalance = self.bm.account.get_balance()
                    elif o.ordtype_bot == 3:
                        self.stop.cancel()

        # adjust limitclose for next candle
        self.limitclose.set_price(price=self.closeprice())
        self.check_timeout()
        self.check_position_closed()

    def all_orders(self):
        return [self.limitopen, self.stop, self.limitclose]
