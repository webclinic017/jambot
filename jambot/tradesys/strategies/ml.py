import warnings
from typing import *

import pandas as pd

from jambot import SYMBOL
from jambot import config as cf
from jambot import display
from jambot import functions as f
from jambot import getlog
from jambot.common import DynConfig
from jambot.ml import models as md
from jambot.tradesys.backtest import BacktestManager
from jambot.tradesys.orders import LimitOrder, MarketOrder, Order, StopOrder
from jambot.tradesys.strategies.base import StrategyBase
from jambot.tradesys.trade import Trade
from jgutils import fileops as jfl
from jgutils import pandas_utils as pu

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

log = getlog(__name__)
warnings.filterwarnings(action='ignore', category=FutureWarning)


class Strategy(StrategyBase, DynConfig):
    log_keys = ['order_offset']

    def __init__(
            self,
            market_on_timeout: bool = False,
            order_offset: float = -0.0006,
            stop_pct: float = None,
            **kw):
        super().__init__(**kw)

        self.use_stops = True if not stop_pct is None else False
        self.market_on_timeout = market_on_timeout
        self.order_offset = order_offset
        self.stop_pct = stop_pct

    def to_dict(self) -> List[str]:
        return super().to_dict() \
            + ['order_offset']

    @property
    def log_items(self) -> Dict[str, Any]:
        return super().log_items \
            | dict(
                order_offset=self.order_offset,
                market_on_timeout=self.market_on_timeout)

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
            price=self.c.close,
            side=side,
            name=name,
            trade=trade)

    def limit_order(
            self,
            price: float,
            side: int,
            offset: float,
            name: str) -> LimitOrder:
        """Create limit order with offset to enter or exit
        """

        limit_price = f.get_price(pnl=offset, price=price, side=side, tick_size=self.symbol.tick_size)

        # NOTE timeout params are kinda arbitrary
        if name == 'open':
            # offset limit_close by eg $1. NOTE not dry, also in Order.adjust_price
            limit_price += -2 * self.symbol.tick_size * side
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
            offset=offset,
            timeout=timeout,
            name=f'limit_{name}',
            trail_close=offset)

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
                order = self.limit_close(target_price, side, self.order_offset).add(trade)
            elif trade.is_pending:
                trade.close()

        return block_next

    def enter_trade(self, side: int, target_price: float) -> None:

        trade = self.make_trade()

        # order = self.market_open(target_price, side)
        order = self.limit_open(target_price, side, self.order_offset).add(trade)
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
            block_next = self.exit_trade(side=c.signal, target_price=c.close)

            if not block_next:
                self.enter_trade(side=c.signal, target_price=c.close)


class StratScorer():
    """Obj to allow scoring only for cross_val test but not train"""
    summary_format = BacktestManager.summary_format | \
        dict(
            test_wt='{:.2f}',
            train_wt='{:.2f}')

    def __init__(self):
        self.p_results = cf.p_data / 'scoring'
        self.p_strats = cf.p_data / 'strats'
        self.reset()

    def reset(self):
        self.m_train = dict(final=True, max=True)
        self.runs = {}
        jfl.clean_dir(self.p_results)
        jfl.clean_dir(self.p_strats)

    def show_summary(self, dfs: List[pd.DataFrame], scores: dict = None) -> None:
        """Show summary df of all backtest runs

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            summary dfs for each cv run
        scores : dict
            ModelManager cross_val scores to show with run results
        """
        import jambot.utils.styles as st

        # bms = list(self.runs.values())  # BacktestManagers
        # if bms:
        #     fmt = bms[0].summary_format
        # dfs = [bm.df_result for bm in bms]
        # dfs = [jfl.load_pickle(p) for p in self.p_results.glob('*')]

        df = pd.concat(dfs) \
            .sort_values('start') \
            .reset_index(drop=True) \
            .pipe(pu.safe_drop, cols='lev')

        # add in test/train weight scores per run
        if scores:
            data = {k: v for k, v in scores.items() if 'wt' in k}
            df = df.join(pd.DataFrame(data))

        fmt = self.summary_format
        higher = ['dd', 'gpct', 'pnl', 'pnl_rt']
        higher_centered = ['min', 'max', 'final']  # centered at 1.0
        higher_centered_2 = ['test_wt', 'train_wt']

        style = df \
            .style.format(fmt) \
            .pipe(st.bg, subset=higher, higher_better=True) \
            .pipe(st.bg, subset=['tpd'], higher_better=False) \
            .pipe(st.bg, subset=higher_centered_2, higher_better=True) \
            .apply(st.background_grad_center, subset=higher_centered, higher_better=True, center=1.0, vmin=0, vmax=100)

        # style.columns = df.pipe(pu.remove_underscore).columns

        ints = ('good', 'filled', 'total')
        df_tot = df.mean().to_frame().T \
            .astype({k: int for k in ints})

        style_tot = df_tot \
            .style.format(fmt)

        display(style.hide_index())
        display(style_tot.hide_index())

    def score(
            self,
            estimator: 'BaseEstimator',
            x: pd.DataFrame,
            y_true: pd.Series,
            _type: str = 'final',
            regression: bool = False,
            symbol: str = SYMBOL,
            **kw) -> float:
        """Run strategy and return final balance
        - called for test then train for x number of splits
        """

        self.m_train[_type] = not self.m_train[_type]
        if self.m_train[_type]:
            return 0

        # NOTE will need to not use proba for regression
        # NOTE could build proba/predict together like mm.add_predict
        idx = x.index
        startdate = idx[0]
        bm = self.runs.get(startdate, None)

        if bm is None:
            df_pred = x.pipe(md.add_preds_probas, pipe=estimator, regression=regression)

            strat = Strategy.from_config(lev=3, live=True, exch_name='bitmex', symbol=symbol, keep_sym=True, **kw)

            cols = ['open', 'high', 'low', 'close', 'signal']
            bm = BacktestManager(startdate=startdate, strat=strat, df=df_pred[cols]).run(prnt=False)
            self.runs[startdate] = bm

            # save df result to disk so can be used with multithreading
            df_res = bm.df_result
            jfl.save_pickle(df_res, p=self.p_results, name=id(df_res))

            # save strat data to be returned from cross_val
            estimator.cv_data = dict(
                startdate=bm.startdate,
                df_result=bm.df_result,
                df_balance=strat.wallet.df_balance,
                df_trades=strat.df_trades())

        wallet = bm.strat.wallet
        if _type == 'final':
            return wallet.balance  # final balance
        elif _type == 'max':
            return wallet.max


def make_strat(**kw) -> Strategy:
    """Initialize strategy with default config

    Returns
    -------
    Strategy
    """
    m = dict(lev=3, market_on_timeout=True, exch_name='bitmex') | kw
    return Strategy(**m)
