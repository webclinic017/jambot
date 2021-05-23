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

        f.set_self(vars())

    def on_attach(self):
        """Market close last trade at end of session"""
        self.parent.end_session.connect(self.exit_trade)

    def exit_trade(self):
        """Market close trade"""
        trade = self.trade

        if not trade is None:
            trade.market_close()
            # trade.cancel_open_orders()
            trade.close()

    def market_open(self, price: float, side: int, qty: int = None, name: str = 'market_open') -> 'MarketOrder':
        """Create market order to enter"""
        if qty is None:
            qty = self.wallet.available_quantity(price=price)

        return MarketOrder(
            symbol=self.symbol,
            qty=qty * side,
            name=name)

    def market_open_late(self, order):
        # NOTE this is v messy still
        # limit order will have stepped forward one timestep from trade
        self.broker.cancel(order)

        trade = self.trade
        order = self.market_open(price=self.c.Close, side=order.side, name='market_open_late')

        print(f'market_open_late - wallet balance: {self.wallet.balance:.3f}, c.Close: {self.c.Close}')

        trade.add_order(order)

    def limit_open(self, price: float, side: int, offset: float) -> 'LimitOrder':
        """Create limit order with offset to enter"""
        limit_price = f.get_price(pnl=offset, entry_price=price, side=side)
        qty = self.wallet.available_quantity(price=limit_price)

        order = LimitOrder(
            symbol=self.symbol,
            qty=qty * side,
            price=limit_price,
            timeout=5,
            name='limit_open')

        order.timedout.connect(self.market_open_late)
        # print('limit added')
        return order

    def enter_trade(self, side: int, target_price: float):

        trade = self.make_trade()
        # qty = self.wallet.available_quantity(price=target_price)

        # order = self.market_open(target_price, side)
        order = self.limit_open(target_price, side, -0.006)

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


class StratScorer():
    """Obj to allow scoring only for cross_val test but not train"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.m_train = dict(final=True, max=True)
        self.runs = {}

    def show_summary(self):
        """Show summary df of all backtest runs"""
        fmt = list(self.runs.values())[0].summary_format

        higher = ['min', 'max', 'final', 'drawdown', 'good_pct']

        dfs = [bm.df_result for bm in self.runs.values()]
        df = pd.concat(dfs) \
            .reset_index(drop=True)

        style = df \
            .style.format(fmt) \
            .pipe(sf.bg, subset=higher, higher_better=True) \
            .pipe(sf.bg, subset=['tpd'], higher_better=False)

        df_tot = df.mean().to_frame().T
        style_tot = df_tot \
            .style.format(fmt)

        display(style)
        display(style_tot)

    def score(self, estimator, x, y_true, _type='final', regression=False, n_smooth=6, **kw):
        """Run strategy and return final balance
        - called for test then train for x number of splits
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
            # print(f'final: {wallet.balance:.2f}')
            return wallet.balance  # final balance
        elif _type == 'max':
            # print(f'max: {wallet.max:.2f}')
            return wallet.max
