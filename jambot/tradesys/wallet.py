from datetime import timedelta as delta

import numpy as np
import pandas as pd

from jambot import Num, display, dt
from jambot import functions as f
from jambot import getlog
from jambot.exchanges.exchange import SwaggerExchange
from jambot.tradesys.base import DictRepr, Observer
from jambot.tradesys.enums import OrderType, TradeSide
from jambot.tradesys.exceptions import InsufficientBalance
from jambot.tradesys.orders import Order
from jambot.tradesys.symbols import Symbol

log = getlog(__name__)


class Txn(DictRepr):
    """Represents a change in base currency balance"""

    def __init__(self, timestamp: dt, balance_pre: float, delta: float):
        self.timestamp = timestamp
        self.balance_pre = balance_pre
        self.delta = delta

    @property
    def balance_post(self):
        return self.balance_pre + self.delta

    @property
    def delta_pct(self):
        """Percent change in account for this transaction"""
        return self.delta / self.balance_pre

    def to_dict(self):
        return dict(
            timestamp=self.timestamp,
            balance_pre=f'{self.balance_pre:.3f}',
            delta=f'{self.delta:+.3f}',
            pnl=f'{self.delta_pct:+.3f}')


class Wallet(Observer):
    """Class to manage transactions and balances for single asset
    - NOTE in future will expand this to have overall Portfolio manage multiple wallets
    """
    exch_fees = dict(
        bitmex=(0.0001, -0.0005),
        bybit=(0.00025, -0.00075),
        binance=(0.00025, -0.00075))  # binance not accurate, just for convenience

    def __init__(
            self,
            symbol: Symbol,
            exch_name: str = 'bitmex',
            df_funding: pd.DataFrame = None,
            **kw):
        super().__init__(**kw)
        self.reset()
        self._total_balance_margin = None  # live trading, comes from exch
        self.prec_balance = 8  # wallet precision (base currency, eg XBT or USDT)
        self._lev = 3
        self.symbol = symbol
        self.exch_name = exch_name
        self.df_funding = df_funding  # TODO track funding trades
        self._df_ci_monthly = None  # type: pd.DataFrame

        self.maker_fee, self.taker_fee = self.exch_fees[exch_name]

    def reset(self):
        self._balance = 1  # base instrument, eg XBT
        self._max = 0
        self._min = 1
        self._min_balance = 0.01
        self.txns = []
        self._qty = 0  # number of open qty
        self.filled_orders = []
        self.price = 0

    def step(self):
        """Check if need to pay funding fee, if funding df given"""

        if not self.df_funding is None and self.c.Index in self.df_funding.index:
            fund_rate = self.df_funding.loc[self.c.Index, 'funding_rate']
            self._balance += self.funding_fee(self.qty, self.price, fund_rate)

    @property
    def side(self) -> TradeSide:
        return TradeSide(np.sign(self.qty))

    @property
    def qty(self) -> Num:
        return self._qty

    @property
    def qty_opp(self) -> Num:
        """Negative of current wallet qty (used to close)"""
        return self.qty * -1

    @property
    def is_zero(self) -> bool:
        """If wallet has zero contracts open"""
        return self.qty == 0

    @property
    def lev(self):
        return self._lev

    @lev.setter
    def lev(self, val: float):
        self._lev = val

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    @property
    def num_txns(self):
        return len(self.txns)

    @property
    def balance(self) -> float:
        """Current wallet balance in base pair, always keep above 0"""
        if self._balance < self._min_balance:
            self._balance = self._min_balance

        return self._balance

    @balance.setter
    def balance(self, balance: float):
        """Update internal balance"""
        self._balance = balance

        if self._balance < self._min:
            self._min = self._balance
        elif self._balance > self._max:
            self._max = self._balance

    @property
    def total_balance_margin(self) -> float:
        """Get current balance plus unrealized pnl gain/loss to know max possible contracts for next trade"""
        if self._total_balance_margin is None:
            # backtesting
            return max(
                self.balance +
                self.get_profit(
                    exit_qty=self.qty * -1,
                    entry_price=self.price,
                    exit_price=self.c.close),
                self._min_balance)
        else:
            # live trading
            return self._total_balance_margin

    @staticmethod
    def funding_fee(qty: Num, price: float, funding_rate: float) -> float:
        """Get funding fee to pay for contracts at price
        - if negative, shorts pay longs
        - on bitmex fee is displayed as "fee rate", already calculated on side of position
        - fee = (contracts / entry_price) * funding_rate
        - NOTE may need to round differently for different symbols
        - Funding fees not significant, ignoring for now

        Parameters
        ----------
        qty : int
            contracts
        price : float
            position entry price
        funding_rate : float

        Returns
        -------
        float
            funding fee
        """
        if price == 0:
            log.warning('price=0')
            return 0.0

        return round((qty / price) * funding_rate * -1, 8)

    def set_exchange_data(self, exch: SwaggerExchange) -> None:
        """Adjust current balance/upnl to match available on exchange

        Parameters
        ----------
        exch : SwaggerExchange
            exchange obj
        """
        # Margin Balance = Wallet Balance + uPNL
        # bitmex does some extra small maths, just multiply by 0.99
        self._total_balance_margin = exch.total_balance_margin * 0.99 * exch.pct_balance

    def fill_order(self, order: 'Order'):
        """Perform transcation of order, modify balance, current price/quantity"""

        price, qty = self.price, self.qty
        qty_pre, price_pre = qty, price

        # adjust current wallet price/quantity
        # NOTE this doesn't currently handle sells which are bigger than current position

        if not order.side * self.side == -1:
            # increasing position, update price

            # check balance available
            avail_qty = self.available_quantity(price=order.price)
            used_qty = self.qty + order.qty

            if used_qty > avail_qty:
                raise InsufficientBalance(self.balance, self.qty, order.qty, avail_qty, order)

            self.price = self.adjust_price(price, order.price, qty, order.qty)

        else:
            # decreasing position, update balance
            delta = self.get_profit(order.qty, price_pre, order.price)
            self.add_transaction(delta)

        # print(f'order qty: {order.qty:+,.{self.symbol.prec_qty}f}, \
        # qty after fill: {self._qty:+,.{self.symbol.prec_qty}f}')
        self._qty += order.qty

        if self.qty == 0:
            self.price = 0

        # fees
        fee = self.calc_fee(order.qty, order.price, order.order_type)
        order.fee = fee
        self._balance += fee

        self.filled_orders.append(order)
        order.fill()

    def add_transaction(self, delta: float):
        """Add transaction record to ledger

        Parameters
        ----------
        delta : float
            delta balance in base inst
        """
        txn = Txn(
            timestamp=self.timestamp,
            balance_pre=self.balance,
            delta=delta)

        self.txns.append(txn)

        self.balance += delta

    def calc_fee(self, qty: Num, price: float, order_type: OrderType) -> float:
        """Calculate fee in base instrument
        - NOTE fees in this way not exact, but accurate to first 3 significant digits
        - FIXME fails for perpetual!!

        Parameters
        ----------
        qty : int
            quantity of contracts
        price : float
            price order executed at
        order_type : OrderType
            Limit orders get rebate, Market/Stop orders pay taker fee

        Returns
        -------
        float
            fee
        """
        # log.info(f'qty: {order.qty:+.0f}, px: {order.price:.0f}, fee: {fee:+.8f}, bal: {self._balance:.6f}')
        fee_rate = self.maker_fee if order_type == OrderType.LIMIT else self.taker_fee
        if self.symbol.is_inverse:
            # inverse perpetual
            fee = (abs(qty) / price) * fee_rate
        else:
            # linear usdt
            fee = abs(qty) * price * fee_rate

        return fee

    def available_quantity(self, price: float) -> Num:
        """Max available quantity to purchase of qty in base pair (eg XBT)

        Parameters
        ----------
        price : float
            price in quote instrument (eg USD)

        Returns
        -------
        int
            quantity of qty
        """
        if self.symbol.is_inverse:
            qty = self.total_balance_margin * self.lev * price
        else:
            qty = self.total_balance_margin * self.lev / price

        # TODO this might make backtesting annoying, might need to start w larger balance for non inverse
        # maybe only do this if live trading?
        # return f.round_down(n=abs(qty), nearest=self.symbol.lot_size)
        return round(qty, self.symbol.prec + 2)

    def adjust_price(
            self,
            price: float,
            order_price: float,
            qty: Num,
            order_qty: Num) -> float:
        """Calculate adjusted basis price when new order filled
        - used for entry/exit price

        Parameters
        ----------
        price : float
            current wallet price
        order_price : float
            current filled order price
        qty : Num
            current wallet quantity
        order_qty : Num
            current filled order quantity

        Returns
        -------
        float
            adjusted price
        """
        return round((price * qty + order_price * order_qty) / (qty + order_qty), self.symbol.prec)

    def get_profit(self, exit_qty: Num, entry_price: float, exit_price: float):
        """Calculate profit in base instrument (XBT)
        - NOTE this will change with different exchange and different base currency

        Parameters
        ----------
        exit_qty : Num
            quantity of qty to sell (decrease position)
        entry_price : float
        exit_price : float

        Returns
        -------
        float
            profit in base instrument
        """
        if 0 in (entry_price, exit_price):
            return 0.0  # first trades wont have entry/exit price yet

        if self.symbol.is_inverse:
            # inverse perpetual
            profit = -1 * exit_qty * (1 / entry_price - 1 / exit_price)
        else:
            # USDT perpetual
            profit = -1 * exit_qty * (exit_price - entry_price)

        return round(profit, self.prec_balance)

    def drawdown(self):
        """Calculate maximum value drawdown during backtest period

        Returns
        -------
        tuple[float, str] : (drawdown, drawdates string)
        """
        drawdown = 0.
        max_seen = 1
        txmax_seen = self.txns[0]
        txhigh, txlow = None, None

        for txn in self.txns:
            val = txn.balance_pre
            if val > max_seen:
                max_seen = val
                txmax_seen = txn

            curdraw = 1 - val / max_seen
            if curdraw > drawdown:
                drawdown = curdraw
                txlow = txn
                txhigh = txmax_seen

        # create string showing max drawdown period
        fmt = '%Y-%m-%d'
        drawdates = '{:.2f} ({}) - {:.2f} ({})'.format(
            txhigh.balance_pre,
            dt.strftime(txhigh.timestamp, fmt),
            txlow.balance_pre,
            dt.strftime(txlow.timestamp, fmt))

        return drawdown * -1, drawdates

    def get_percent_change(self, balance, change):
        return f.percent(change / balance)

    def print_summary(self, period='month'):
        # TODO: make this include current month, also make str rep
        data = []
        cols = ['Period', 'AcctBalance', 'Change', 'PercentChange']

        periodnum = self.get_period_num(self.txns[0].timestamp, period)
        prevTxn = None
        prevBalance = 1
        change = 0.0

        for t in self.txns:
            if self.get_period_num(t.timestamp, period) != periodnum:
                if prevTxn is None:
                    prevTxn = t
                data.append([
                    periodnum,
                    '{:.{prec}f}'.format(t.balance_post, prec=3),
                    round(change, 3),
                    self.get_percent_change(prevBalance, change)
                ])
                prevTxn = t
                prevBalance = t.balance_post
                change = 0
                periodnum = self.get_period_num(t.timestamp, period)

            change += t.amount

        df = pd.DataFrame(data=data, columns=cols)
        display(df)

    def s_pnl_monthly(self, df: pd.DataFrame = None) -> pd.Series:
        """Get series of monthly pnl

        Parameters
        ----------
        df : pd.DataFrame, optional
            df_balance (pass in to avoid recalc), by default None

        Returns
        -------
        pd.Series
            df with monthly timestamps index
        """
        if df is None:
            df = self.df_balance

        s = df.resample(rule='M').last()
        s.loc[s.index.min() + delta(days=-31)] = 1
        s.sort_index(inplace=True)

        return s.pct_change() \
            .set_index(s.index + delta(days=-15)) \
            .dropna()['balance']

    @property
    def df_balance(self) -> pd.DataFrame:
        """df of balance at all transactions"""

        m = {t.timestamp: t.balance_post for t in self.txns}

        return pd.DataFrame \
            .from_dict(
                m,
                orient='index',
                columns=['balance']) \
            .rename_axis('timestamp')

    def plot_balance(self, logy: bool = True, title: str = None) -> None:
        """Show plot of account balance over time with red/blue color depending on slope"""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib import dates as mdates
        from matplotlib.collections import LineCollection
        from matplotlib.colors import rgb2hex
        from seaborn import diverging_palette
        _cmap = diverging_palette(240, 10, n=21, as_cmap=True)

        df = self.df_balance

        y = df.balance
        dy = np.gradient(y)

        # hrs = df.index.to_series().diff().dt.total_seconds() / 3600
        dyy = df.balance.pct_change()  # / hrs

        x = mdates.date2num(df.index.to_pydatetime())
        dx = np.gradient(x)
        dydx = dy / dx

        outer_bound = min(abs(dyy.min()), abs(dyy.max()))
        # norm = plt.Normalize(dydx.min(), dydx.max())
        norm = plt.Normalize(outer_bound * -1, outer_bound)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=_cmap.reversed(), norm=norm)

        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            figsize=(14, 7),
            sharex=True,
            gridspec_kw=dict(height_ratios=(3, 1)))

        fig.suptitle(str(self.symbol))

        # monthly pct change
        s = self.s_pnl_monthly(df=df)

        blue = rgb2hex(_cmap(0))
        red = rgb2hex(_cmap(_cmap.N))
        colors = (s > 0).apply(lambda x: blue if x else red)

        ax2.bar(x=s.index, height=s.values, width=10, color=colors)
        ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0%}'))

        for p in ax2.patches:
            height = p.get_height()
            ax2.text(
                size=12,
                x=p.get_x() + p.get_width() / 2,
                y=height + 0.2,
                s=f'{height:,.0%}',
                ha='center')

        lc.set_array(dydx)
        lc.set_linewidth(2)
        ax1.add_collection(lc)

        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        monthFmt = mdates.DateFormatter('%Y-%m')
        ax1.xaxis.set_major_formatter(monthFmt)

        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

        ax1.grid(axis='y', linewidth=0.3, which='both')
        ax1.grid(axis='x', linewidth=0.3, which='major')
        ax1.autoscale_view()

        plt.xticks(rotation=45)
        plt.tight_layout()

    def print_txns(self):
        data = []
        cols = ['Date', 'AcctBalance', 'Amount', 'PercentChange']
        for t in self.txns:
            data.append([
                '{:%Y-%m-%d %H}'.format(t.timestamp),
                '{:.{prec}f}'.format(t.balance_pre, prec=3),
                '{:.{prec}f}'.format(t.amount, prec=2),
                f.percent(t.percentchange)
            ])

        pd.options.display.max_rows = len(data)
        df = pd.DataFrame(data=data, columns=cols)
        display(df)
        pd.options.display.max_rows = 100

    def get_df(self):
        df = pd.DataFrame(columns=['timestamp', 'Balance', 'PercentChange'])
        for i, t in enumerate(self.txns):
            df.loc[i] = [t.timestamp, t.balance_pre, t.percentchange]
        return df

    @property
    def df_filled_orders(self) -> pd.DataFrame:
        data = [o.to_dict() for o in self.filled_orders]
        return pd.DataFrame.from_dict(data)

    def show_orders(self, last: int = 30) -> None:
        display(self.df_filled_orders.iloc[-last:])

    @property
    def df_ci_monthly(self) -> pd.DataFrame:
        """Get df of daily > monthly compound interest rates per month

        Returns
        -------
        pd.DataFrame
        """
        # just to avoid recalc when logging both weighted and unweighted sharpe ratio
        if not self._df_ci_monthly is None:
            return self._df_ci_monthly

        df = self.df_balance.resample(rule='M').last() \
            .assign(initial=lambda x: x.balance.shift(1))

        # add starting balance for first month
        df.iloc[0, df.columns.get_loc('initial')] = 1

        self._df_ci_monthly = df \
            .assign(
                ci_rate=lambda x: np.vectorize(f.ci_rate)(x.balance, x.index.day, x.initial),
                w_ci_rate=lambda x: x.ci_rate * np.linspace(0.5, 1, len(x)))

        return self._df_ci_monthly

    def sharpe(self, weighted: bool = False) -> float:
        """Calculate sharpe ratio across monthly returns

        Parameters
        ----------
        weighted : bool, optional
            weight more recent months higher, by default False

        Returns
        -------
        float
        """
        df = self.df_ci_monthly
        ci_col = 'ci_rate' if not weighted else 'w_ci_rate'
        return df[ci_col].mean() / df.ci_rate.std()

    def ci_monthly(self, weighted: bool = False) -> float:
        """Calculate avg compound interest across monthly returns

        Parameters
        ----------
        weighted : bool, optional
            weight more recent months higher, by default False

        Returns
        -------
        float
        """
        df = self.df_ci_monthly
        ci_col = 'ci_rate' if not weighted else 'w_ci_rate'
        return df[ci_col].mean()
