from .. import (
    backtest as bt,
    signals as sg,
    functions as f)
from ..backtest import Order

class Strategy(bt.Strategy):
    def __init__(self, speed=(18,18), weight=1, lev=5, emaspeed=(50, 200)):
        super().__init__(weight, lev)
        self.name = 'trend'
        self.enteroffset = 0
        self.speed = speed
        self.lasthigh, self.lastlow = 0, 0
        self.emaspeed = emaspeed

    def init(self, sym):
        self.sym = sym
        df = sym.df
        fast, slow = self.emaspeed[0], self.emaspeed[1]
            
        macd = sg.MACD(df=sym.df, weight=1, fast=fast, slow=slow)
        ema = sg.EMA(df=sym.df, weight=1, fast=fast, slow=slow)
        # emaslope = EMASlope(df=df, weight=1, p=50, slope=5)
        self.conf.add_signal([macd, ema])
        
        self.trend = sg.Trend(df=df, signals=[df.ema_trend], speed=self.speed)

    def get_recent_win_conf(self):
        # TODO: could probs do this better with list/filter
        recentwin = False
        closedtrades = self.tradecount()
        ctOffset = closedtrades - 1 if closedtrades < 3 else 2
        for y in range(closedtrades, closedtrades - ctOffset - 1, -1):
            if self.get_trade(y).pnlfinal > 0.05:
                recentwin = True
        
        recentwinconf = 0.25 if recentwin else 1
        return recentwinconf

    def get_confidence(self, side):
        conf = self.conf.final(side=side, c=self.cdl)

        recentwinconf = self.get_recent_win_conf()
        confidence = recentwinconf if recentwinconf <= 0.5 else conf
        return confidence

    def enter_trade(self, side, entryprice):
        self.trade = self.init_trade(trade=Trade(), side=side, entryprice=entryprice, conf=self.get_confidence(side=side))
        self.trade.check_orders(self.sym.cdl)

    def exit_trade(self):
        self.trade.exit_trade()

    def decide(self, c):
        self.cdl = c
        self.i = c.Index
        pxhigh, pxlow = c.pxhigh, c.pxlow

        # Exit Trade
        if not self.trade is None:
            self.trade.check_orders(c)

            if self.trade.side == 1:
                if c.Low < pxlow and c.Low < self.lastlow:
                    self.exit_trade()                
            else:
                if c.High > pxhigh and c.High > self.lasthigh:
                    self.exit_trade()

            if not self.trade.active:
                self.trade = None
        
        # Enter Trade
        if self.trade is None:
            pxhigh *= (1 + self.enteroffset)
            pxlow *= (1 - self.enteroffset)

            if c.High > pxhigh:
                self.enter_trade(1, pxhigh)
            elif c.Low < pxlow:
                self.enter_trade(-1, pxlow)

        self.lasthigh, self.lastlow = pxhigh, pxlow

    def final_orders(self, u, weight):
        # should actually pass something at the 'position' level, not user?
        lstOrders = []
        c = self.sym.cdl
        side = self.get_side()
        price = c.trend_low if self.status == 1 else c.trend_high
        
        #TODO: use trade's orders now
        # stopclose
        lstOrders.append(Order(
                    price=price,
                    side=-1 * side,
                    contracts=-1 * u.get_position(self.sym.symbolbitmex)['currentQty'],
                    symbol=self.sym.symbolbitmex,
                    name='stopclose',
                    ordtype='Stop',
                    sym=self.sym))

        # stopbuy
        contracts = f.get_contracts(
                        u.totalbalancewallet * weight,
                        self.lev,
                        price,
                        -1 * side,
                        self.sym.altstatus)

        lstOrders.append(Order(
                    price=price,
                    side=-1 * side,
                    contracts=contracts,
                    symbol=self.sym.symbolbitmex,
                    name='stopbuy',
                    ordtype='Stop',
                    sym=self.sym))

        return self.checkfinalorders(lstOrders)
 
class Trade(bt.Trade):
    def __init__(self):
        super().__init__()

    def closeprice(self):
        c = self.cdl
        price = c.pxlow if self.side == 1 else c.pxhigh

        return price

    def enter(self, temp=False):

        contracts = int(self.targetcontracts * self.conf)

        self.stopopen = Order(
                    price=self.entrytarget,
                    side=self.side,
                    contracts=contracts,
                    activate=True,
                    ordtype_bot=4,
                    ordtype='Stop',
                    name='stopopen',
                    trade=self)

        self.stopclose = Order(
                    price=self.closeprice(),
                    side=self.side * -1,
                    contracts=contracts,
                    activate=False,
                    ordtype_bot=2,
                    ordtype='Stop',
                    name='stopclose',
                    execinst='Close',
                    trade=self)

    def check_orders(self, c):
        self.add_candle(c)

        for o in self.orders:
            if o.active and not o.filled:
                o.check(c)

                if o.filled:
                    if o.ordtype_bot == 4:
                        self.filledcontracts = o.contracts
                        self.stopclose.active = True
        
        self.stopclose.price = self.closeprice()
    