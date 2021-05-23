from ..base import Observer, SignalEvent
from ..broker import Broker
from ..trade import Trade
from .__init__ import *


class StrategyBase(Observer):
    def __init__(self, symbol: str, weight=1, lev=5, slippage=0.02, **kw):
        super().__init__(**kw)

        trades = []
        broker = Broker(parent_listener=self)
        wallet = broker.get_wallet(symbol)
        wallet.lev = lev  # NOTE would prefer to init with this

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
        """Total number of trades"""
        return len(self.trades)

    @property
    def num_trades_filled(self):
        """Total number of trades where at least one order was executed"""
        return len([t for t in self.trades if abs(t.qty) > 0])

    @property
    def trade(self) -> 'Trade':
        """Return last (current) trade"""
        return self.trades[-1] if len(self.trades) > 0 else None

    @property
    def tpd(self):
        """Trades per day frequency"""
        return self.num_trades / (self.duration / 24)

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

    @property
    def good_trades(self) -> int:
        return sum([t.is_good for t in self.trades])

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

    def show_trades(self, maxmin=0, first=float('inf'), last=0, df=None):
        import seaborn as sns

        if df is None:
            df = self.df_trades(first=first, last=last)
        style = df.style.hide_index()

        # figs = self.bm.decimal_figs
        # price_format = '{:,.' + str(figs) + 'f}'
        price_format = '{:,.0f}'

        cmap = sns.diverging_palette(10, 240, sep=10, n=20, center='dark', as_cmap=True)
        # style.background_gradient(cmap=cmap, subset=['PnlAcct'], vmin=-0.3, vmax=0.3)

        style \
            .background_gradient(cmap=cmap, subset=['pnl'], vmin=-0.1, vmax=0.1) \
            .format({
                'timestamp': '{:%Y-%m-%d %H}',
                'qty': '{:,}',
                'entry': price_format,
                'exit': price_format,
                #   'Conf': '{:.3f}',
                'pnl': '{:.2%}',
                #   'PnlAcct': '{:.2%}',
                'bal': '{:.2f}'
            })

        display(style)

    def df_trades(self, first=float('inf'), last=0):
        """Return df of trade history"""
        trades = self.trades
        data = [t.dict_stats() for t in trades[last * -1: min(first, len(trades))]]

        return pd.DataFrame \
            .from_dict(data=data) \
            .assign(profitable=lambda x: x.pnl > 0)
