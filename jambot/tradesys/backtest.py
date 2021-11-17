from typing import *

import pandas as pd

from jambot import display
from jambot import functions as f
from jambot.tradesys.base import Clock, SignalEvent
from jambot.tradesys.strategies.base import StrategyBase
from jambot.utils.mlflow import MlflowLoggable


class BacktestManager(Clock, MlflowLoggable):
    """Organize and run strategies over dataframe with signals"""

    # Dict to use for styling summary df
    fmt_date = '{:%Y-\u2060%m-\u2060%d}'  # no line breaks
    summary_format = dict(
        start=fmt_date,
        end=fmt_date,
        dur='{:,.0f}',
        min='{:.2f}',
        max='{:.1f}',
        final='{:.1f}',
        dd='{:.1%}',
        dur_gpm='{:.1f}',
        pnl='{:.1%}',
        pnl_rt='{:.1f}',
        pnl_mm='{:.1%}',
        tpd='{:.2f}',
        good='{:,.0f}',
        filled='{:,.0f}',
        total='{:,.0f}',
        gpct='{:.0%}',
        gfpct='{:.0%}')

    # rename data cols for display
    m_conv = dict(
        pct_good_filled='gpct',
        pct_good_total='gfpct',
        pnl_med_monthly='pnl_mm',
        drawdown='dd',
        pnl_final_ratio='pnl_rt',
        dur_good_pnl_mean='dur_gpm'
    )

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
        self.strat = strat
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
        style = self.df_result \
            .rename(columns=self.m_conv) \
            .style \
            .format(self.summary_format) \
            .hide_index()

        display(style)

    @property
    def log_items(self) -> dict:
        """Get first (only) row of summary df as dict for logging"""
        return self.df_result.iloc[0].to_dict()

    def log_dfs(self) -> List[Tuple[pd.DataFrame, str]]:
        cols = ['target', 'y_pred', 'proba_long', 'rolling_proba', 'signal']
        return dict(df=self.df[cols].dropna(), name='df_pred', keep_index=True)

    @property
    def df_result(self):
        """df of backtest results"""
        strat = self.strat
        df_trades = strat.df_trades()
        a = strat.wallet

        drawdown, drawdates = a.drawdown()
        pnl = df_trades.pnl.sum()

        # TODO pnl diff btwn signal and actual fill

        data = dict(
            start=self.df.index[0],
            end=self.df.index[-1],
            dur=self.df.shape[0],
            dur_good_pnl_mean=df_trades.query('pnl > 0').dur.mean(),
            min=a.min,
            max=a.max,
            final=a.balance,
            drawdown=drawdown,
            pnl=pnl,
            pnl_med_monthly=a.s_pnl_monthly().median(),
            pnl_final_ratio=a.balance / pnl,
            tpd=strat.tpd,
            good=strat.good_trades,
            filled=strat.num_trades_filled,
            total=strat.num_trades
        )

        # only count "good" as pct of filled trades, ignore unfilled
        return pd.DataFrame. \
            from_dict(data, orient='index').T \
            .assign(
                pct_good_filled=lambda x: x.good / x.filled,
                pct_good_total=lambda x: x.good / x.total)
