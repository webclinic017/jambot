from ..base import Observer, SignalEvent
from ..broker import Broker
from ..trade import Trade
from .__init__ import *


class StrategyBase(Observer):
    def __init__(self, symbol: str, weight=1, lev=5, slippage=0.02, **kw):
        super().__init__(**kw)
        i = 1
        # status = 0
        # self.weight = weight
        # entry_price = 0
        # exit_price = 0
        # maxspread = 0.1
        # self.slippage = slippage
        # self.lev = lev
        # unfilledtrades = 0
        # timeout = float('inf')
        trades = []

        broker = Broker(parent_listener=self)
        wallet = broker.get_wallet(symbol)

        f.set_self(vars())

    def step(self):
        pass

    @property
    def side(self):
        """Use current/last trade to determine side"""
        trade = self.trade
        return trade.side if not trade is None else None

    @property
    def num_trades(self):
        return len(self.trades)

    @property
    def trade(self) -> 'Trade':
        """Return last (current) trade"""
        return self.trades[-1] if len(self.trades) > 0 else None

    def add_trade(self, trade: 'Trade'):
        self.trades.append(trade)

    def make_trade(self) -> 'Trade':
        """Create trade object with wallet/broker"""

        trade = Trade(
            parent_listener=self,
            symbol=self.symbol,
            broker=self.broker)

        self.add_trade(trade)

        return trade

    def last_trade(self):
        return self.trades[self.tradecount() - 1]

    def get_trade(self, i):
        numtrades = self.tradecount()
        if i > numtrades:
            i = numtrades
        return self.trades[i - 1]

    def good_trades(self):
        count = 0
        for t in self.trades:
            if t.is_good():
                count += 1
        return count

    def get_side(self):
        status = self.status
        if status == 0:
            return 0
        elif status < 0:
            return - 1
        elif status > 0:
            return 1

    def bad_trades(self):
        return list(filter(lambda x: x.pnlfinal <= 0, self.trades))

    def print_trades(self, maxmin=0, first=float('inf'), last=0, df=None):
        import seaborn as sns

        if df is None:
            df = self.df_trades(first=first, last=last)
        style = df.style.hide_index()

        figs = self.bm.decimal_figs
        price_format = '{:,.' + str(figs) + 'f}'

        cmap = sns.diverging_palette(10, 240, sep=10, n=20, center='dark', as_cmap=True)
        style.background_gradient(cmap=cmap, subset=['Pnl'], vmin=-0.1, vmax=0.1)
        style.background_gradient(cmap=cmap, subset=['PnlAcct'], vmin=-0.3, vmax=0.3)

        style.format({'Timestamp': '{:%Y-%m-%d %H}',
                      'Contracts': '{:,}',
                      'Entry': price_format,
                      'Exit': price_format,
                      'Conf': '{:.3f}',
                      'Pnl': '{:.2%}',
                      'PnlAcct': '{:.2%}',
                      'Bal': '{:.2f}'})
        display(style)

    def df_trades(self, first=float('inf'), last=0):
        """Return df of trade history"""
        data = []
        trades = self.trades
        cols = ['N', 'Timestamp', 'Sts', 'Dur', 'Entry', 'Exit',
                'Contracts', 'Conf', 'Pnl', 'PnlAcct', 'Bal']  # 'Market',

        for t in trades[last * -1: min(first, len(trades))]:
            data.append([
                t.tradenum,
                t.candles[0].Index,
                t.status,
                t.duration(),
                t.entry_price,
                t.exit_price,
                # t.exit_order().ordtype_str(),
                t.filledcontracts,
                t.conf,
                t.pnlfinal,
                t.pnl_acct(),
                t.exitbalance])

        return pd.DataFrame.from_records(data=data, columns=cols) \
            .assign(profitable=lambda x: x.Pnl > 0)
