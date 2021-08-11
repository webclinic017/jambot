from .. import backtest as bt
from .. import functions as f
from .. import signals as sg
from ..backtest import Order
from . import trend


class Strategy(trend.Strategy):
    def __init__(self, speed=(18, 18), emaspeed=(200, 800)):
        super().__init__(speed=speed, emaspeed=emaspeed)
        self.name = 'trendopen'

    def init(self, bm):
        self.bm = bm
        df = bm.df
        fast, slow = self.emaspeed[0], self.emaspeed[1]

        macd = sg.MACD(df=bm.df, weight=1, fast=fast, slow=slow)
        ema = sg.EMA(df=bm.df, weight=1, fast=fast, slow=slow)
        self.conf.add_signal([macd, ema])

        self.trend = sg.Trend(df=df, offset=6, signals=[df.ema_trend], speed=self.speed)

    def enter_trade(self, side, c):
        self.trade = self.init_trade(trade=Trade(), side=side, entry_price=c.close, conf=self.get_confidence(side=side))
        self.trade.add_candle(c)

    def exit_trade(self):
        t, c = self.trade, self.cdl
        t.marketclose.fill(c=c, price=c.close)
        t.exit_trade()

    def decide(self, c):
        self.cdl = c
        self.i = c.Index
        pxhigh, pxlow = c.pxhigh, c.pxlow

        # Exit Trade
        if not self.trade is None:
            t = self.trade
            t.add_candle(c)

            if t.side == 1:
                if c.close < pxlow:
                    self.exit_trade()
            else:
                if c.close > pxhigh:
                    self.exit_trade()

            if not t.active:
                self.trade = None

        # Enter Trade
        if self.trade is None:
            pxhigh *= (1 + self.enteroffset)
            pxlow *= (1 - self.enteroffset)

            if c.close > pxhigh:
                self.enter_trade(side=1, c=c)
            elif c.close < pxlow:
                self.enter_trade(side=-1, c=c)

        # self.lasthigh, self.lastlow = pxhigh, pxlow


class Trade(bt.Trade):
    def __init__(self):
        super().__init__()

    def enter(self, temp=False):

        qty = int(self.targetcontracts * self.conf)

        self.marketopen = Order(
            price=self.entrytarget,
            side=self.side,
            qty=qty,
            activate=True,
            ordtype_bot=5,
            ordtype='Market',
            name='marketopen',
            trade=self)

        self.marketclose = Order(
            price=self.entrytarget,
            side=self.side * -1,
            qty=qty,
            activate=False,
            ordtype_bot=6,
            ordtype='Market',
            name='marketclose',
            exec_inst='close',
            trade=self)

        self.marketopen.fill(c=self.cdl)
