from jambot.tradesys.base import Clock, SignalEvent
from jambot.tradesys.strategies.base import StrategyBase

from .__init__ import *


class BacktestManager(Clock):
    """Organize and run strategies over dataframe with signals"""

    def __init__(
            self,
            strat: 'StrategyBase',
            df: pd.DataFrame,
            startdate: str,
            symbol: str = 'XBTUSD',
            u=None,
            **kw):

        super().__init__(df=df, **kw)

        self.end_session = SignalEvent()

        # if not isinstance(strats, list): strats = [strats]

        # if row is None:
        #     dfsym = pd.read_csv(cf.p_res / 'symbols.csv')
        #     dfsym = dfsym[dfsym['symbol'] == symbol]
        #     row = list(dfsym.itertuples())[0]

        # self.row = row
        # symbolshort = row.symbolshort
        # urlshort = row.urlshort
        # symbolbitmex = row.symbolbitmex
        # altstatus = bool(row.altstatus)
        # decimal_figs = row.decimal_figs
        # tradingenabled = True
        # self.partial = partial

        # self.symbol = symbol
        startdate = f.check_date(startdate)

        # actual backtest, not just admin info
        if not startdate is None:

            df = df[df.index >= startdate]

            # if df is None:
            #     df = db.get_df(symbol=symbol, startdate=startdate, daterange=daterange)

            # if partial:
            #     if u is None:
            #         u = live.User()
            #     df = u.append_partial(df)

            # startrow = df.index.get_loc(startdate)

            # for strat in self.strats:
            # strat.init(bm=self, df=df)

        self.attach_listener(strat)

        f.set_self(vars())

    def step(self):
        """Top level doesn't need to do anything"""
        pass

    def run(self, prnt=True) -> None:
        """Top level step function"""
        super().run()
        self.end_session.emit()

    @property
    def summary_format(self):
        """Dict to use for styling summary df"""
        return dict(
            start='{:%Y-%m-%d}',
            end='{:%Y-%m-%d}',
            dur='{:,.0f}',
            min='{:.3f}',
            max='{:.3f}',
            final='{:.3f}',
            drawdown='{:.1%}',
            tpd='{:.2f}',
            good='{:,.0f}',
            total='{:,.0f}',
            good_pct='{:.0%}'
        )

    def print_final(self):
        """Style backtest summary df"""
        style = self.df_result.style \
            .format(self.summary_format) \
            .hide_index()

        display(style)

    @property
    def df_result(self):
        """df of backtest results"""
        strat = self.strat
        a = strat.wallet

        drawdown, drawdates = a.drawdown()

        data = dict(
            # symbol=self.symbol,
            start=self.df.index[0],
            end=self.df.index[-1],
            dur=self.df.shape[0],
            min=a.min,
            max=a.max,
            final=a.balance,
            drawdown=drawdown,
            # period=drawdates,
            tpd=strat.tpd,
            lev=strat.lev,
            good=strat.good_trades,
            filled=strat.num_trades_filled,
            total=strat.num_trades
        )

        # only count "good" as pct of filled trades, ignore unfilled
        return pd.DataFrame. \
            from_dict(data, orient='index').T \
            .assign(good_pct=lambda x: x.good / x.filled)

    # def write_csv(self):
    #     self.df.to_csv('dfout.csv')
    #     self.account.get_df().to_csv('df2out.csv')

    # def expected_orders(self):
    #     # don't use this yet.. maybe when we have combined strats?

    #     expected = []
    #     for strat in self.strats:
    #         for order in strat.final_orders():
    #             expected.append(order)

    #     return expected
