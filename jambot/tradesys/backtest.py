import pandas as pd

from jambot import display
from jambot import functions as f
from jambot.tradesys.base import Clock, SignalEvent
from jambot.tradesys.strategies.base import StrategyBase
from jambot.utils.mlflow import MLFlowLoggable


class BacktestManager(Clock, MLFlowLoggable):
    """Organize and run strategies over dataframe with signals"""

    # Dict to use for styling summary df
    summary_format = dict(
        start='{:%Y-%m-%d}',
        end='{:%Y-%m-%d}',
        dur='{:,.0f}',
        min='{:.2f}',
        max='{:.1f}',
        final='{:.1f}',
        dd='{:.1%}',
        pnl='{:.1%}',
        pnl_rt='{:.1f}',
        tpd='{:.2f}',
        good='{:,.0f}',
        filled='{:,.0f}',
        total='{:,.0f}',
        gpct='{:.0%}')

    def __init__(
            self,
            strat: 'StrategyBase',
            df: pd.DataFrame,
            startdate: str,
            **kw):

        super().__init__(df=df, **kw)

        self.end_session = SignalEvent()

        startdate = f.check_date(startdate)

        # actual backtest, not just admin info
        if not startdate is None:
            df = df[df.index >= startdate]

        self.attach_listener(strat)

        f.set_self(vars())

    def step(self):
        """Top level doesn't need to do anything"""
        pass

    def run(self, prnt: bool = False, _log: bool = False) -> 'BacktestManager':
        """Top level step function

        Parameters
        ----------
        prnt : bool
            print summary stats at end of run
        _log : bool
            log to mlflow, default False

        Returns
        -------
        BacktestManager
            self
        """
        self.strat.wallet.reset()
        super().run()
        self.end_session.emit()

        if prnt:
            self.print_final()
            self.strat.wallet.plot_balance(logy=True)

        if _log:
            self.log_mlflow()

        return self

    def print_final(self):
        """Style backtest summary df"""
        style = self.df_result.style \
            .format(self.summary_format) \
            .hide_index()

        display(style)

    @property
    def log_items(self) -> dict:
        return self.df_result.iloc[0].to_dict() \
            | self.strat.log_items

    @property
    def df_result(self):
        """df of backtest results"""
        strat = self.strat
        df_trades = strat.df_trades()
        a = strat.wallet

        drawdown, drawdates = a.drawdown()
        pnl = df_trades.pnl.sum()

        data = dict(
            # symbol=self.symbol,
            start=self.df.index[0],
            end=self.df.index[-1],
            dur=self.df.shape[0],
            min=a.min,
            max=a.max,
            final=a.balance,
            dd=drawdown,
            pnl=pnl,
            pnl_rt=a.balance / pnl,
            # period=drawdates,
            tpd=strat.tpd,
            # lev=strat.lev,
            good=strat.good_trades,
            filled=strat.num_trades_filled,
            total=strat.num_trades
        )

        # only count "good" as pct of filled trades, ignore unfilled
        return pd.DataFrame. \
            from_dict(data, orient='index').T \
            .assign(gpct=lambda x: x.good / x.filled)
