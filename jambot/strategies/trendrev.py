from .. import (
    backtest as bt,
    signals as sg,
    functions as f)
from ..backtest import Order, Strategy, Trade

class Strategy(Strategy):
    def __init__(self, speed=(8,8), weight=1, norm=(0.004, 0.024), lev=5):
        super().__init__(weight, lev)
        self.name = 'trendrev'
        self.stoppercent = -0.03
        self.speed = speed
        self.norm = norm
        self.lasthigh, self.lastlow = 0, 0
        self.timeout = 40
        self.slippage = 0

    def init(self, sym):
        self.sym = sym
        df = sym.df
        self.a = self.sym.account
        
        self.vty = sg.Volatility(df=df, weight=1, norm=self.norm)

        macd = sg.MACD(df=df, weight=1)
        ema = sg.EMA(df=df, weight=1)
        emaslope = sg.EMASlope(df=df, weight=1, p=50, slope=5)
        self.conf.add_signal(signals=[macd, ema, emaslope],
                            trendsignals=[ema])

        self.trend = sg.Trend(df=df, signals=[df.ema_trend, df.ema_slope], speed=self.speed)

    def exit_trade(self):
        t = self.trade
        c = self.cdl

        if not t.stopped and t.limitopen.filled and not t.limitclose.filled:
            t.stop.cancel()
            t.limitclose.fill(c=c, price=c.Close)
            self.unfilledtrades += 1
            
        t.exit_trade()
        self.trade = None
    
    def enter_trade(self, side, entryprice):
        self.trade = self.init_trade(trade=Trade(), side=side, entryprice=entryprice)
        t = self.trade
        t.check_orders(self.sym.cdl)
        if not t.active: self.exit_trade()
        
    def decide(self, c):
        self.cdl = c
        self.i = c.Index
        pxhigh, pxlow = c.pxhigh, c.pxlow

        # if we exit a trade and limitclose isnt filled, limitopen may wait and fill in next candles, but current trade gets 0 profit. Need to exit properly.

        # Exit Trade
        if not self.trade is None:
            t = self.trade

            if t.side == 1:
                if c.High > max(pxhigh, self.lasthigh):
                    t.active = False
            else:
                if c.Low < min(pxlow, self.lastlow):
                    t.active = False

            t.check_orders(c)
            if not t.active: self.exit_trade()

            # need to make trade active=False if limitclose is filled?
            # order that candle moves is important
            # TODO: need enter > exit > enter all in same candle

        # Enter Trade
        enterhigh, enterlow = self.lasthigh, self.lastlow
        # enterhigh, enterlow = pxhigh, pxlow
        if self.trade is None:
            if c.High > enterhigh:
                self.enter_trade(-1, enterhigh)
            elif c.Low < enterlow:
                self.enter_trade(1, enterlow)

        self.lasthigh, self.lastlow = pxhigh, pxlow

    def final_orders(self, u, weight):
        symbol = self.sym.symbolbitmex
        balance = u.balance() * weight
        pos = bt.Position(contracts=u.get_position(symbol)['currentQty'])
        pos.add_order(u.get_orders(symbol=symbol, manualonly=True))

        t_prev = self.trades[-2]
        t_current = self.trades[-1]
        t_current.rescale_orders(balance=balance)

        prevclose = t_prev.limitclose
        limitopen = t_current.limitopen
        stopclose = t_current.stop
        limitclose = t_current.limitclose

        # PREVCLOSE - Check if position is still open with correct side/contracts
        if (prevclose.marketfilled 
        and pos.side() == t_prev.side 
        and t_current.duration() <= 4):
            pos.add_order(Order(
                            contracts=pos.contracts * -1,
                            ordtype='Market',
                            name='marketclose',
                            trade=t_prev))

        # OPEN
        if limitopen.filled:
            if (limitopen.marketfilled 
            and pos.contracts == 0
            and t_current.duration() == 4):
                pos.add_order(Order(
                                contracts=limitopen.contracts,
                                ordtype='Market',
                                name='marketopen',
                                trade=t_current))
        else:
            pos.add_order(limitopen) # Only till max 4 candles into trade

        # STOP
        if not stopclose.filled:
            pos.add_order(stopclose)

        # CLOSE - current
        if not pos.contracts == 0:            
            if t_current.timedout:
                pos.add_order(Order(
                                contracts=limitclose.contracts,
                                ordtype='Market',
                                name='marketclose',
                                trade=t_current))
            else:
                pos.add_order(limitclose)

            if stopclose.contracts == 0 or not stopclose in pos.orders:
                f.discord(msg='Error: no stop for current position', channel='err')

        # NEXT - Init next trade to get next limitopen and stop
        c = self.cdl
        t_next = self.init_trade(
                        side=t_current.side * -1,
                        entryprice=(c.pxhigh if t_current.side == 1 else c.pxlow),
                        balance=balance,
                        temp=True,
                        trade=Trade())

        pos.add_order(t_next.limitopen)
        pos.add_order(t_next.stop)

        return pos.final_orders()

class Trade(Trade):
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
        return round(price * (1 + self.enteroffset * self.side), self.sym.decimalfigs)
    
    def enter(self):
        c = self.strat.cdl
        self.stoppercent = self.strat.stoppercent
        # self.enteroffset = self.strat.vty.final(i=c.Index)
        self.enteroffset = c.norm_ema

        limitbuyprice = self.entrytarget * (1 + self.enteroffset * self.side * -1)
        limitcloseprice = self.closeprice()
        self.stoppx = f.get_price(self.stoppercent, limitbuyprice, self.side)

        contracts = int(self.targetcontracts * self.conf)

        self.limitopen = Order(
                    price=limitbuyprice,
                    side=self.side,
                    contracts=contracts,
                    activate=True,
                    ordtype_bot=1,
                    ordtype='Limit',
                    name='limitopen',
                    trade=self)

        self.stop = Order(
                    price=self.stoppx,
                    side=self.side * -1,
                    contracts=contracts,
                    activate=False,
                    ordtype_bot=2,
                    ordtype='Stop',
                    name='stop',
                    reduce=True,
                    trade=self)

        self.limitclose = Order(
                    price=limitcloseprice,
                    side=self.side * -1,
                    contracts=contracts,
                    activate=False,
                    ordtype_bot=3,
                    ordtype='Limit',
                    name='limitclose',
                    reduce=True,
                    trade=self)
    
    def check_position_closed(self):
        if self.limitclose.filled:
            self.active = False
    
    def check_timeout(self):
        if self.duration() >= self.timeout:
            self.timedout = True
            self.active = False
    
    def check_orders(self, c):
        # trade stays active until pxlow is hit, strat controlls
        # filling order sets the trade's actual entryprice
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
                        o.fill(c=c, price=c.Close)

                if o.filled:
                    if o.ordtype_bot == 1:
                        self.filledcontracts = o.contracts
                        delay = 1 if o.marketfilled else 0
                        self.limitclose.activate(c=c, delay=delay)
                        self.stop.activate(c=c, delay=0)
                    elif o.ordtype_bot == 2:
                        self.limitclose.cancel() #make limitclose filledtime be end of trade
                        self.stopped = True
                        self.pnlfinal = f.get_pnl(self.side, self.entryprice, self.exitprice)
                        self.exitbalance = self.sym.account.get_balance()
                    elif o.ordtype_bot == 3:
                        self.stop.cancel()
            
        # adjust limitclose for next candle
        self.limitclose.set_price(price=self.closeprice())
        self.check_timeout()
        self.check_position_closed()

    def all_orders(self):
        return [self.limitopen, self.stop, self.limitclose]