from .. import backtest as bt
from .. import functions as f
from .. import signals as sg
from .__init__ import *
from .base import StrategyBase


class Strategy(StrategyBase):
    def __init__(self, weight=1, lev=5, **kw):
        super().__init__(weight, lev, **kw)
        self.name = 'SFP'
        self.minswing = 0.05
        self.stypes = dict(high=1, low=-1)

    def init(self, bm=None, df=None):

        if not bm is None:
            self.bm = bm
            self.df = bm.df
            df = self.df
            self.a = self.bm.account
        elif not df is None:
            self.df = df

        offset = 6
        period_base = 48  # 48, 96, 192

        for i in range(3):
            period = period_base * 2 ** i

            df[f'sfp_high{i}'] = df.High.rolling(period).max().shift(offset)
            df[f'sfp_low{i}'] = df.Low.rolling(period).min().shift(offset)

        ema = sg.EMA(weight=1)
        df = df.pipe(ema.add_signal)
        self.df = df
        # self.bm.df = df

    def check_tail(self, side, cdl):
        return True if cdl.tailsize(side=side) / cdl.size() > self.minswing else False

    def check_swing(self, side, swingval, cdl):
        c = cdl.row
        px_max = c.High if side == 1 else c.Low

        if (side * (px_max - swingval) > 0 and
                side * (c.Close - swingval) < 0):
            return True

    def is_swingfail(self, c=None, i=None):
        if c is None:
            if i is None:
                i = len(self.df) - 1

            c = self.df.iloc[i]

        cdl = Candle(row=c)
        self.cdl = cdl
        stypes = self.stypes
        sfp = []

        # Swing High - need to check with multiple periods, largest to smallest?
        # only count further back if high/low is higher
        for k in stypes:
            side = stypes[k]
            prevmax = float('-inf') * side

            for i in range(3):
                swing = f'{k}{i}'
                # TODO: this is sketch
                if 'pandas.core.frame.Pandas' in str(type(c)):
                    swingval = c._asdict()[f'sfp_{swing}']
                else:
                    swingval = c[f'sfp_{swing}']

                if (side * (swingval - prevmax) > 0 and
                    self.check_swing(side=side, swingval=swingval, cdl=cdl) and
                        self.check_tail(side=side, cdl=cdl)):
                    sfp.append(dict(name=swing, price=swingval))

                prevmax = swingval

        return sfp

    def enter_trade(self, side, price, c):
        self.trade = self.init_trade(trade=Trade(), side=side, entry_price=price)
        self.trade.add_candle(c)

    def exit_trade(self, price):
        self.trade.exit_(price=price)
        self.trade = None

    def decide(self, c):
        stypes = self.stypes

        # EXIT Trade
        if not self.trade is None:
            t = self.trade
            t.add_candle(c)
            if t.duration() == 12:
                self.exit_trade(price=c.Close)

        # ENTER Trade
        # if swing fail, then enter in opposite direction at close
        # if multiple swing sides, enter with candle side

        if self.trade is None:
            sfps = self.is_swingfail(c=c)

            if sfps:
                cdl = self.cdl
                m = {}
                for k in stypes.keys():
                    m[k] = len(list(filter(lambda x: k in x['name'], sfps)))

                m2 = {k: v for k, v in m.items() if v > 0}
                if len(m2) > 1:
                    swingtype = cdl.side()
                else:
                    swingtype = stypes[list(m2)[0]]

                self.enter_trade(side=swingtype * -1, price=c.Close, c=c)


class Trade(bt.Trade):
    def __init__(self):
        super().__init__()

    def exit_(self, price):
        self.marketclose.price = price  # so that not 'marketfilled'
        self.marketclose.fill(c=self.cdl)
        self.exit_trade()

    def enter(self):
        self.marketopen = Order(
            price=self.entrytarget,
            side=self.side,
            qty=self.targetcontracts,
            activate=True,
            ordtype_bot=5,
            ordtype='Market',
            name='marketopen',
            trade=self)

        self.marketclose = Order(
            price=self.entrytarget,
            side=self.side * -1,
            qty=self.targetcontracts,
            activate=True,
            ordtype_bot=6,
            ordtype='Market',
            name='marketclose',
            trade=self)

        self.marketopen.fill(c=self.cdl)
