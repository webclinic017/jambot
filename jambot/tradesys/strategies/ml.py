from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from ... import signals as sg
from ... import sklearn_helper_funcs as sf
from ..backtest import Backtest
from ..orders import LimitOrder, MarketOrder, StopOrder
from ..trade import Trade
from .__init__ import *
from .base import StrategyBase

log = getlog(__name__)


class Strategy(StrategyBase):
    def __init__(
            self,
            min_proba=0.5,
            min_agree=0,
            stoppercent=None,
            min_proba_enter=0.8,
            num_disagree=0,
            min_agree_pct=0.8,
            regression=False,
            **kw):
        super().__init__(**kw)

        use_stops = True if not stoppercent is None else False
        split_val = 0 if regression else 0.5

        # Trade = TradeML
        trades = []

        f.set_self(vars())

    def init(self, sym, df):
        # a = sym.account
        icol_y_pred = df.columns.get_loc('y_pred')
        f.set_self(vars())

    def exit_trade(self):
        """Market close trade"""
        self.trade.market_close()

    def enter_trade(self, side: int, target_price: float):

        trade = self.make_trade()

        qty = self.wallet.available_quantity(leverage=self.lev, price=target_price)

        order = MarketOrder(
            qty=qty * side,
            name='market_open')

        trade.add_order(order)

    def step(self, c):
        # self.cdl = c
        # df = self.df
        t = self.trade

        # signal_side = c.y_pred
        # signal_side = c.y_pred # psar strat
        # assign current side based on rolling proba

        signal_side = 1 if c.rolling_proba > self.split_val else -1

        # cur_price = c.Close
        # if not t is None:
        #     side = t.side
        #     t.check_orders(c) # broker checks all orders
        # else:
        #     side = 0

        # track current trade side
        # check if signal side changed (y_pred)
        if not signal_side == 0 and not self.side == signal_side:
            self.exit_trade()
            self.enter_trade(side=signal_side, target_price=self.c.Close)

            # proba = {
            #     -1: c.proba_short,
            #     1: c.proba_long}.get(signal_side)

            # # check if prev predictions agree to minimize excessive switching
            # i = df.index.get_loc(c.Index)
            # s = df.iloc[i - self.min_agree: i, self.icol_y_pred] \
            #     .pipe(lambda s: s[s != 0])

            # if all(s == signal_side): # or proba > self.min_proba_enter:

            #     if proba > self.min_proba:
            #         self.exit_trade(exit_price=cur_price)
            #         self.enter_trade(side=signal_side, entry_price=cur_price)

        # close final trade at last candle to see pnl
        # TODO do this with a "final candle" signal or similar
        # if c.Index == df.index[-1]:
        #     self.exit_trade(exit_price=cur_price)


# class TradeML(bt.TradeBase):
#     def __init__(self, **kw):
#         super().__init__(**kw)

#     def step(self, c):

#         if self.limitopen.filled:
#             self.close()

#     def enter(self):

#         contracts = int(self.targetcontracts * self.conf)

#         offset_pct = 0.0025
#         limitbuyprice = self.entrytarget * (1 + offset_pct * self.side * -1)

#         self.limitopen = Order(
#             price=limitbuyprice,
#             side=self.side,
#             contracts=contracts,
#             ordtype_bot=1,
#             ordtype='Limit',
#             name='limitopen',
#             trade=self)

#         # .activate(c=self.strat.cdl, timeout=4)

#         # MARKET OPEN
#         # self.marketopen = Order(
#         #     price=self.entrytarget,
#         #     side=self.side,
#         #     contracts=contracts,
#         #     activate=True,
#         #     ordtype_bot=5,
#         #     ordtype='Market',
#         #     name='marketopen',
#         #     trade=self)

#         # self.marketopen.fill(price=self.entrytarget, c=self.strat.cdl)

#         # STOP
#         if self.strat.use_stops:
#             self.stoppercent = self.strat.stoppercent
#             self.stoppx = f.get_price(self.stoppercent, self.entrytarget, self.side)
#             self.stop = Order(
#                 price=self.stoppx,
#                 side=self.side * -1,
#                 contracts=contracts,
#                 ordtype_bot=2,
#                 ordtype='Stop',
#                 name='stop',
#                 reduce=True,
#                 trade=self)

#     # def market_close(self, price):
#     #     self.marketclose = Order(
#     #         price=price,
#     #         side=self.side * -1,
#     #         contracts=self.marketopen.contracts * -1,
#     #         activate=True,
#     #         ordtype_bot=6,
#     #         ordtype='Market',
#     #         name='marketclose',
#     #         trade=self)

#     #     self.marketclose.fill(price=price, c=self.strat.cdl)

#     def check_orders(self, c):
#         """Check if stop is hit"""
#         # self.add_candle(c)
#         # if not self.strat.use_stops:
#         #     return

#         for o in self.orders:
#             if not o.filled and o.is_active(c=c) and o.ordtype in ('Stop', 'Limit'):
#                 o.check(c=c)

#                 if o.filled:
#                     # print(
#                     #     f'side: {self.side}',
#                     #     f'entryprice: {self.entryprice}',
#                     #     f'stoppx: {o.price}',
#                     #     f'exitprice: {self.exitprice}'
#                     # )
#                     if o.ordtype == 'Limit':
#                         pass
#                     elif o.ordtype == 'Stop':
#                         # NOTE not sure if this should be here
#                         self.stopped = True
#                         self.pnlfinal = f.get_pnl(self.side, self.entryprice, self.exitprice)
#                         self.exitbalance = self.sym.account.get_balance()


class StratScorer():
    """Obj to allow scoring only for cross_val test but not train"""

    def __init__(self):
        self.is_train = True
        self.m_train = dict(final=True, max=True)
        self.runs = {}

    def score(self, estimator, x, y_true, _type='final', regression=False, n_smooth=6, **kw):
        """Run strategy and return final balance
        - called for test then train for x number of splits...
        """

        self.m_train[_type] = not self.m_train[_type]
        if self.m_train[_type]:
            return 0

        # NOTE will need to not use proba for regression
        # NOTE could build proba/predict together like mm.add_predict
        idx = x.index
        sym = self.runs.get(idx[0], None)

        if sym is None:
            rolling_col = 'proba_long' if not regression else 'y_pred'

            df_pred = x \
                .assign(y_pred=estimator.predict(x)) \
                .join(sf.df_proba(df=x, model=estimator)) \
                .pipe(sg.add_ema, p=n_smooth, c=rolling_col, col='rolling_proba') \
                # .assign(rolling_proba=lambda x: x[rolling_col].rolling(n_smooth).mean()) \

            strat = Strategy(
                lev=3,
                slippage=0,
                regression=regression)

            kw_args = dict(
                symbol='XBTUSD',
                startdate=idx[0])

            cols = ['Open', 'High', 'Low', 'Close', 'y_pred', 'proba_long', 'rolling_proba']
            sym = Backtest(**kw_args, strat=strat, df=df_pred[cols])
            sym.decide_full(prnt=False)

            self.runs[idx[0]] = sym

        if _type == 'final':
            print(f'final: {sym.account.balance:.2f}')
            return sym.account.balance  # final balance
        elif _type == 'max':
            print(f'max: {sym.account.max:.2f}')
            return sym.account.max
