from .. import (
    backtest as bt,
    signals as sg,
    functions as f)
from ..backtest import Order, Strategy, Trade

class Strategy(Strategy):
    def __init__(self, min_proba=0.5, min_agree=0, stoppercent=-0.03, min_proba_enter=0.8, num_disagree=0, min_agree_pct=0.8, use_stops=False, **kw):
        super().__init__(**kw)
        f.set_self(vars())

    def init(self, sym):
        df = sym.df
        a = sym.account
        icol_y_pred = df.columns.get_loc('y_pred')
        f.set_self(vars())

    def exit_trade(self, exit_price):
        t = self.trade
        c = self.cdl
        
        if not t is None:
            if not t.stopped:
                t.market_close(price=exit_price)

            t.exit_trade()

        self.trade = None

    def enter_trade(self, side, entry_price):
        self.trade = self.init_trade(trade=Trade(), side=side, entryprice=entry_price, conf=1)

    def decide(self, c):
        self.cdl = c
        df = self.df
        t = self.trade
        # self.i = c.Index
        
        cur_side = c.y_pred
        cur_price = c.Close
        if not t is None:
            side = t.side
            t.check_orders(c)
        else:
            side = 0

        # track current trade side
        # check if side changed (y_pred)
        if not cur_side == 0 and not side == cur_side:

            proba = {
                -1: c.proba_short,
                1: c.proba_long}.get(cur_side)
            
            # check if prev predictions agree to minimize excessive switching
            i = df.index.get_loc(c.Index)
            s = df.iloc[i - self.min_agree: i, self.icol_y_pred] \
                .pipe(lambda s: s[s != 0])

            if all(s == cur_side): # or proba > self.min_proba_enter:

                if proba > self.min_proba:
                    self.exit_trade(exit_price=cur_price)
                    self.enter_trade(side=cur_side, entry_price=cur_price)
    
class Trade(Trade):
    def __init__(self):
        super().__init__()
    
    # def init(self, price, **kw):
    #     super().init(price=price, entryprice=price, **kw)

    def enter(self):

        contracts = int(self.targetcontracts * self.conf)

        # MARKET OPEN
        self.marketopen = Order(
            price=self.entrytarget,
            side=self.side,
            contracts=contracts,
            activate=True,
            ordtype_bot=5,
            ordtype='Market',
            name='marketopen',
            trade=self)
        
        self.marketopen.fill(price=self.entrytarget, c=self.strat.cdl)

        # STOP
        if self.strat.use_stops:
            self.stoppercent = self.strat.stoppercent
            self.stoppx = f.get_price(self.stoppercent, self.entrytarget, self.side)
            self.stop = Order(
                price=self.stoppx,
                side=self.side * -1,
                contracts=contracts,
                activate=True,
                ordtype_bot=2,
                ordtype='Stop',
                name='stop',
                reduce=True,
                trade=self)
    
    def market_close(self, price):
        self.marketclose = Order(
            price=price,
            side=self.side * -1,
            contracts=self.marketopen.contracts * -1,
            activate=True,
            ordtype_bot=6,
            ordtype='Market',
            name='marketclose',
            trade=self)
        
        self.marketclose.fill(price=price, c=self.strat.cdl)
    
    def check_orders(self, c):
        """Check if stop is hit"""
        self.add_candle(c)
        if not self.strat.use_stops:
            return

        for o in self.orders:
            if not o.filled and o.ordtype == 'Stop':
                o.check(c=c)

                if o.filled:
                    # print(
                    #     f'side: {self.side}',
                    #     f'entryprice: {self.entryprice}',
                    #     f'stoppx: {o.price}',
                    #     f'exitprice: {self.exitprice}'
                    # )

                    # NOTE not sure if this should be here
                    self.stopped = True
                    self.pnlfinal = f.get_pnl(self.side, self.entryprice, self.exitprice)
                    self.exitbalance = self.sym.account.get_balance()