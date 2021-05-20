from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from ... import signals as sg
from ... import sklearn_helper_funcs as sf
from ..backtest import BacktestManager
from ..orders import LimitOrder, MarketOrder, StopOrder
from ..trade import Trade
from .__init__ import *
from .base import StrategyBase

log = getlog(__name__)


class Strategy(StrategyBase):
    def __init__(
            self,
            symbol: str = 'XBTUSD',
            min_proba=0.5,
            min_agree=0,
            stoppercent=None,
            min_proba_enter=0.8,
            num_disagree=0,
            min_agree_pct=0.8,
            regression=False,
            **kw):
        super().__init__(symbol=symbol, **kw)

        use_stops = True if not stoppercent is None else False
        # split_val = 0 if regression else 0.5

        # Trade = TradeML
        trades = []

        f.set_self(vars())

    # def init(self, bm, df):
    #     # a = bm.account
    #     icol_y_pred = df.columns.get_loc('y_pred')
    #     f.set_self(vars())

    def exit_trade(self):
        """Market close trade"""
        trade = self.trade

        if not trade is None:
            trade.market_close()

    def enter_trade(self, side: int, target_price: float):

        trade = self.make_trade()

        qty = self.wallet.available_quantity(leverage=self.lev, price=target_price)

        order = MarketOrder(
            symbol=self.symbol,
            qty=qty * side,
            name='market_open')

        trade.add_order(order)

    def step(self):
        # self.cdl = c
        # df = self.df
        t = self.trade
        c = self.c

        # signal_side = c.y_pred
        # signal_side = c.y_pred # psar strat
        # assign current side based on rolling proba

        # signal_side = 1 if self.c.rolling_proba > self.split_val else -1

        if c.signal in (1, -1):
            self.exit_trade()
            self.enter_trade(side=c.signal, target_price=c.Close)

        # close final trade at last candle to see pnl
        # TODO do this with a "final candle" signal or similar
        # if c.Index == df.index[-1]:
        #     self.exit_trade(exit_price=cur_price)


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
        bm = self.runs.get(idx[0], None)

        if bm is None:
            rolling_col = 'proba_long' if not regression else 'y_pred'

            df_pred = x \
                .assign(y_pred=estimator.predict(x)) \
                .join(sf.df_proba(df=x, model=estimator)) \
                .pipe(sg.add_ema, p=n_smooth, c=rolling_col, col='rolling_proba') \
                .assign(
                    signal=lambda x: np.sign(np.diff(np.sign(x.rolling_proba - 0.5), prepend=np.array([0])))) \
                .fillna(0)
            # .assign(rolling_proba=lambda x: x[rolling_col].rolling(n_smooth).mean()) \

            strat = Strategy(
                lev=3,
                slippage=0,
                regression=regression)

            kw_args = dict(
                symbol='XBTUSD',
                startdate=idx[0])

            cols = ['Open', 'High', 'Low', 'Close', 'signal']
            bm = BacktestManager(**kw_args, strat=strat, df=df_pred[cols])
            bm.run(prnt=False)

            self.runs[idx[0]] = bm

        wallet = bm.strat.wallet
        if _type == 'final':
            print(f'final: {wallet.balance:.2f}')
            return wallet.balance  # final balance
        elif _type == 'max':
            print(f'max: {wallet.max:.2f}')
            return wallet.max
