import warnings

from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from ... import signals as sg
from ... import sklearn_utils as sk
from ..backtest import BacktestManager
from ..orders import LimitOrder, MarketOrder, Order, StopOrder
from ..trade import Trade
from .__init__ import *
from .base import StrategyBase

log = getlog(__name__)
warnings.filterwarnings(action='ignore', category=FutureWarning)


class Strategy(StrategyBase):
    def __init__(
            self,
            symbol: str = 'XBTUSD',
            min_proba=0.5,
            min_agree=0,
            stop_pct=None,
            min_proba_enter=0.8,
            num_disagree=0,
            min_agree_pct=0.8,
            regression=False,
            market_on_timeout: bool = False,
            **kw):
        super().__init__(symbol=symbol, **kw)

        use_stops = True if not stop_pct is None else False
        # split_val = 0 if regression else 0.5

        f.set_self(vars())

    def limit_open(self, *args, **kw) -> LimitOrder:
        return self.limit_order(name='open', *args, **kw)

    def limit_close(self, *args, **kw) -> LimitOrder:
        return self.limit_order(name='close', *args, **kw)

    def market_open(self, *args, **kw) -> MarketOrder:
        return self.market_order(name='open', *args, **kw)

    def market_close(self, *args, **kw) -> MarketOrder:
        return self.market_order(name='close', *args, **kw)

    def market_order(
            self,
            price: float,
            side: int,
            qty: int = None,
            name: str = 'open',
            trade: 'Trade' = None) -> MarketOrder:
        """Create market order to enter"""
        if name == 'open':
            if qty is None:
                qty = self.wallet.available_quantity(price=price) * side
        else:
            qty = self.wallet.qty_opp

        if qty == 0:
            return

        order = MarketOrder(
            symbol=self.symbol,
            qty=qty,
            name=f'market_{name}').add(trade)

    def market_late(self, order: Order, name: str) -> None:
        """Create market order after triggered by order expiry

        Parameters
        ----------
        order : Order
            order triggering late open
        name : str
            open/close
        """
        side = order.side if name == 'open' else order.side_opp

        trade = order.parent
        order = self.market_order(
            price=self.c.Close,
            side=side,
            name=name,
            trade=trade)

    def limit_order(
            self,
            price: float,
            side: int,
            offset: float,
            name: str) -> LimitOrder:
        """Create limit order with offset to enter or exit"""

        # if hasattr(self.c, 'pred_max'):
        #     minmax_col = {-1: 'pred_max', 1: 'pred_min'}.get(side)
        #     offset = abs(getattr(self.c, minmax_col)) * -1 * 0.25

        # minmax_col = {-1: 'target_max', 1: 'target_min'}.get(side)
        # offset_true = abs(getattr(self.c, minmax_col)) * -1  #* 0.5

        limit_price = f.get_price(pnl=offset, entry_price=price, side=side)

        if name == 'open':
            qty = self.wallet.available_quantity(price=limit_price) * side
            timeout = 6
        else:
            # close any open contracts
            timeout = 4
            qty = self.wallet.qty_opp

        order = LimitOrder(
            symbol=self.symbol,
            qty=qty,
            price=limit_price,
            timeout=timeout,
            name=f'limit_{name}')

        if self.market_on_timeout or name == 'close':
            order.timedout.connect(lambda order, name=name: self.market_late(order, name=name))

        return order

    def exit_trade(self, side: int = None, target_price: float = None) -> bool:
        """Market/Limit close trade"""
        block_next = False
        trade_prev = self.get_trade(-2)  # close trade_-2
        trade = self.trade  # trade_-1

        if not trade_prev is None:
            # if older trade with same side is still open, just need to cancel
            if trade_prev.is_pending:
                trade_prev.close()

            elif trade_prev.is_open:
                # treat this as the new "current" trade, move to top of stack
                trade_prev.cancel_open_orders()
                self.add_trade(trade_prev)
                block_next = True

        if not trade is None:
            if trade.is_open:
                order = self.limit_close(target_price, side, -0.0005).add(trade)
            elif trade.is_pending:
                trade.close()

        return block_next

    def enter_trade(self, side: int, target_price: float) -> None:

        trade = self.make_trade()

        # order = self.market_open(target_price, side)
        order = self.limit_open(target_price, side, -0.0005).add(trade)
        trade_prev = self.get_trade(-2)

        if not trade_prev is None:
            trade_prev.closed.connect(order.adjust_max_qty)

        if self.use_stops:
            stop_order = StopOrder.from_order(order=order, stop_pct=self.stop_pct)
            stop_order.filled.connect(trade.close)
            trade.add_order(stop_order)

    def step(self):
        c = self.c

        if c.signal in (1, -1):
            block_next = self.exit_trade(side=c.signal, target_price=c.Close)

            if not block_next:
                self.enter_trade(side=c.signal, target_price=c.Close)


class StratScorer():
    """Obj to allow scoring only for cross_val test but not train"""

    def __init__(self, n_smooth: int = 6):
        self.reset()
        f.set_self(vars())

    def reset(self):
        self.m_train = dict(final=True, max=True)
        self.runs = {}

    def show_summary(self):
        """Show summary df of all backtest runs"""
        fmt = list(self.runs.values())[0].summary_format

        higher = ['drawdown', 'good_pct']
        higher_centered = ['min', 'max', 'final']  # centered at 1.0

        dfs = [bm.df_result for bm in self.runs.values()]
        df = pd.concat(dfs) \
            .reset_index(drop=True)

        style = df \
            .style.format(fmt) \
            .pipe(sk.bg, subset=higher, higher_better=True) \
            .pipe(sk.bg, subset=['tpd'], higher_better=False) \
            .apply(sk.background_grad_center, subset=higher_centered, higher_better=True, center=1.0) \

        ints = ('lev', 'good', 'filled', 'total')
        df_tot = df.mean().to_frame().T \
            .astype({k: int for k in ints})

        style_tot = df_tot \
            .style.format(fmt)

        display(style)
        display(style_tot)

    def score(self, estimator, x, y_true, _type='final', regression=False, **kw):
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
                .join(sk.df_proba(df=x, model=estimator)) \
                .pipe(sg.add_ema, p=self.n_smooth, c=rolling_col, col='rolling_proba') \
                .assign(signal=lambda x: sk.proba_to_signal(x.rolling_proba)) \
                # .assign(rolling_proba=lambda x: x[rolling_col].rolling(n_smooth).mean()) \

            strat = Strategy(
                lev=3,
                slippage=0,
                regression=regression,
                market_on_timeout=False)

            kw_args = dict(
                symbol='XBTUSD',
                startdate=idx[0])

            cols = ['Open', 'High', 'Low', 'Close', 'signal']
            bm = BacktestManager(**kw_args, strat=strat, df=df_pred[cols])
            bm.run(prnt=False)

            self.runs[idx[0]] = bm

        wallet = bm.strat.wallet
        if _type == 'final':
            return wallet.balance  # final balance
        elif _type == 'max':
            return wallet.max
