from .__init__ import *
from .base import Observer, SignalEvent
from .strategies.base import StrategyBase

# from .database import db
# from .tradesys.trade import Trade
# from .tradesys.broker import Broker


class Backtest(Observer):
    def __init__(self, strat: 'StrategyBase', df: pd.DataFrame, startdate: str, u=None, **kw):

        # if not isinstance(strats, list): strats = [strats]

        # if row is None:
        #     dfsym = pd.read_csv(f.topfolder / 'data/symbols.csv')
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
            #     df = db.get_dataframe(symbol=symbol, startdate=startdate, daterange=daterange)

            # if partial:
            #     if u is None:
            #         u = live.User()
            #     df = u.append_partial(df)

            # startrow = df.index.get_loc(startdate)

            # for strat in self.strats:
            strat.init(sym=self, df=df)

        f.set_self(vars())

    def step(self):
        """Top level doesn't need to do anything"""
        pass

    def run(self, prnt=True) -> None:
        """Top level step function"""
        df = self.df

        if prnt:
            idx = df.index
            print(f'Test range: {idx[0]} - {idx[-1]}')

        self.attach(self.strat, c=None)

        for c in df.itertuples():
            self._step(c)
            # self.strat.decide(c)

        if self.partial:
            self.strat.trades[-1].partial = True

    def print_final(self):
        style = self.df_result().style.hide_index()
        style.format({'Min': '{:.3f}',
                      'Max': '{:.3f}',
                      'Final': '{:.3f}',
                      'Drawdown': '{:.2%}'})
        display(style)

    def df_result(self):
        a = self.account
        strat = self.strat

        drawdown, drawdates = a.drawdown()

        data = {
            'symbol': [self.symbol],
            'Min': [a.min],
            'Max': [a.max],
            'Final': [a.balance],
            'Drawdown': [drawdown],
            'Period': [drawdates],
            'Goodtrades': ['{}/{}/{}'.format(strat.good_trades(), strat.tradecount(), strat.unfilledtrades)]}

        return pd.DataFrame(data=data)

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
