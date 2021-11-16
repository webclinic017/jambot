from typing import *

import pandas as pd

from jambot import display
from jambot import functions as f
from jambot import getlog
from jambot.tradesys.base import Observer
from jambot.tradesys.broker import Broker
from jambot.tradesys.trade import Trade
from jambot.utils.mlflow import MLFlowLoggable

log = getlog(__name__)


class StrategyBase(Observer, MLFlowLoggable):
    def __init__(
            self,
            symbol: str,
            weight: int = 1,
            lev: int = 3,
            slippage: float = 0.02,
            live: bool = False,
            exch_name: str = 'bitmex',
            **kw):
        super().__init__(**kw)

        _trades = []
        self._broker = Broker(parent_listener=self, symbol=symbol, exch_name=exch_name)
        self.wallet = self._broker.get_wallet(symbol)
        self.wallet.lev = lev

        f.set_self(vars())

    def on_attach(self):
        """Market close last trade at end of session (if not live trading)"""
        if not self.live:
            self.get_parent('BacktestManager').end_session.connect(self.final_market_close)

    def final_market_close(self):
        t = self.get_trade(-1, open_only=True)
        if not t is None:
            t.market_close()

    def step(self):
        pass

    def to_dict(self) -> dict:
        m = dict(exch=self.exch_name)

        if not self.parent is None:
            m_fmt = self.parent.summary_format
            _m = self.parent.df_result.iloc[0].to_dict()
            m |= {k: m_fmt[k].format(v) if k in m_fmt else v for k, v in _m.items()}

        return m

    @property
    def log_items(self) -> Dict[str, Any]:
        return dict(
            symbol=self.symbol,
            lev=self.lev)

    @property
    def df(self) -> pd.DataFrame:
        """Convenience to get parent df"""
        return self.parent.df if not self.parent is None else None

    @property
    def broker(self) -> Broker:
        """Broker"""
        return self._broker

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
    def trades(self) -> List[Trade]:
        return self._trades

    @property
    def trade(self) -> Union[Trade, None]:
        """Return last (current) trade"""
        return self.trades[-1] if len(self.trades) > 0 else None

    def get_trade(self, i: int, open_only: bool = False) -> Union[Trade, None]:
        """Get trade by index

        Parameters
        ----------
        i : int
            trade index to return
        open_only : bool, default False
            only index trades with open orders

        Returns
        -------
        Union[Trade, None]
            Trade if exists
        """
        if open_only:
            trades = [t for t in self.trades if t.is_open]
        else:
            trades = self.trades

        try:
            return trades[i]
        except IndexError:
            return

    @property
    def tpd(self):
        """Trades per day frequency"""
        return self.num_trades / (self.duration / 24)

    def add_trade(self, trade: Trade) -> None:
        """Append trade to list of self.trades"""
        self.trades.append(trade)

    def make_trade(self) -> Trade:
        """Create trade object with wallet/broker"""

        trade = Trade(
            parent_listener=self,
            symbol=self.symbol,
            broker=self.broker)

        self.add_trade(trade)
        trade.trade_num = len(self.trades)

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

    def show_trades(
            self,
            maxmin: int = 0,
            first: int = float('inf'),
            last: int = 20,
            df: pd.DataFrame = None) -> None:

        from jambot.utils.styles import _cmap

        df = df or self.df_trades(first=first, last=last)
        style = df.style.hide_index()

        # figs = self.bm.decimal_figs
        # price_format = '{:,.' + str(figs) + 'f}'
        price_format = '{:,.0f}'

        style \
            .background_gradient(cmap=_cmap.reversed(), subset=['pnl', 'pnl_acct'], vmin=-0.1, vmax=0.1) \
            .format({
                'ts': '{:%Y-%m-%d %H:%M}',
                'qty': '{:+,}',
                'entry': price_format,
                'exit': price_format,
                'pnl': '{:.2%}',
                'pnl_acct': '{:.2%}',
                'bal': '{:.2f}'})

        display(style)

    def df_trades(self, first: int = float('inf'), last: int = 0) -> pd.DataFrame:
        """Return df of trade history

        Parameters
        ----------
        first : int, optional
            only first n rows, by default all
        last : int, optional
            only last n rows, by default all

        Returns
        -------
        pd.DataFrame
        """
        trades = self.trades
        data = [t.dict_stats() for t in trades[last * -1: min(first, len(trades))]]

        return pd.DataFrame \
            .from_dict(data=data) \
            .assign(profitable=lambda x: x.pnl > 0) \
            .drop_duplicates(subset=['t_num'], keep='first')

    def trade_dist(self) -> None:
        """Show trade % win/loss histogram"""
        df = self.df_trades()
        df.pnl.plot(kind='hist', bins=50)

    def err_summary(self, last: int = 30) -> None:
        self.broker.show_orders(last=last)
        self.wallet.show_orders(last=last)
        self.show_trades(last=last)
        print('Time failed: ', self.timestamp)
