from .__init__ import *
from .base import Observer
from .enums import TradeSide
from .orders import Order

log = getlog(__name__)


class Txn(object):
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
        return round(self.delta / self.balance_pre, 3)

    def printTxn(self):
        pass


class Wallet(Observer):
    """Class to manage transactions and balances for single asset
    - NOTE in future will expand this to have overall Portfolio manage multiple wallets
    """

    def __init__(self, symbol: str):
        super().__init__()
        _balance = 1  # base instrument, eg XBT
        _default_balance = _balance
        _min_balance = 0.01
        precision = 8
        _max = 0
        _min = _balance
        txns = []
        # _df_balance = None
        # self.symbol = symbol
        qty = 0  # number of open contracts
        price = 0  # entry price of current position
        f.set_self(vars())

    def step(self):
        pass

    @property
    def side(self):
        return TradeSide(np.sign(self.qty))

    @property
    def balance(self):
        """Current wallet balance in base pair, always keep above 0"""
        if self._balance < self._min_balance:
            self._balance = self._min_balance

        return self._balance

    @balance.setter
    def balance(self, balance: float):
        """Update internal balance and add transaction record"""
        # self.add_transaction(delta=delta)
        self._balance = balance

        if self._balance < self._min:
            self._min = self._balance
        elif self._balance > self._max:
            self._max = self._balance

    @property
    def num_txns(self):
        return len(self.txns)

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

    def available_quantity(self, leverage: int, price: float) -> int:
        """Max available quantity to purchase of contracts in base pair (eg XBT)

        Parameters
        ----------
        leverage : int
            leverage to use
        price : float
            price in quote instrument (eg USD)

        Returns
        -------
        int
            quantity of contracts
        """
        qty = self.balance * leverage * price
        return int(qty)

    def reset(self):
        self._balance = self._default_balance

    def fill_order(self, order: 'Order'):
        """Perform transcation of order, modify balance, current price/quantity"""

        price, qty = self.price, self.qty
        qty_pre, price_pre = qty, price

        # adjust current wallet price/quantity
        # NOTE this doesn't currently handle sells which are bigger than current position

        if not order.side * self.side == -1:
            # increasing position, update price
            self.price = (price * qty + order.price * order.qty) / (qty + order.qty)
        else:
            # decreasing position, update balance
            delta = self.get_profit(order.qty, price_pre, order.price)
            self.add_transaction(delta)

        self.qty += order.qty

        if self.qty == 0:
            self.price = 0

        order.fill()

    def get_profit(self, exit_qty: int, entry_price: float, exit_price: float):
        """Calculate profit in base instrument (XBT)
        - NOTE this will change with different exchange and different base currency

        Parameters
        ----------
        exit_qty : int
            quantity of contracts to sell (decrease position)
        entry_price : float
        exit_price : float

        Returns
        -------
        float
            profit in base instrument
        """
        profit = -1 * exit_qty * (1 / entry_price - 1 / exit_price)
        return round(profit, self.precision)

    # def modify(self, xbt, timestamp):
    #     txn = Txn()
    #     txn.amount = xbt
    #     txn.timestamp = timestamp
    #     txn.balance_pre = self.balance
    #     txn.percentchange = round((xbt / self.balance), 3)

    #     self.txns.append(txn)
    #     self.balance = self.balance + xbt
    #     txn.balance_post = self.balance

    #     if self.balance > self.max:
    #         self.max = self.balance

    #     if self.balance < self.min:
    #         self.min = self.balance

    # def get_period_num(self, timestamp, period='month'):
    #     timestamp = f.check_date(timestamp)
    #     with f.Switch(period) as case:
    #         if case('month'):
    #             return timestamp.month
    #         elif case('week'):
    #             return timestamp.strftime("%V")

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
        drawdates = '{:.2f} ({}) - {:.2f} ({})'.format(
            txhigh.balance_pre,
            dt.strftime(txhigh.timestamp, f.time_format()),
            txlow.balance_pre,
            dt.strftime(txlow.timestamp, f.time_format()))

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

    @property
    def df_balance(self):
        if self._df_balance is None:
            m = dd(list)

            for t in self.txns:
                m['timestamp'].append(t.timestamp)
                m['balance'].append(t.balance_post)

            self._df_balance = pd.DataFrame.from_dict(m) \
                .set_index('timestamp')

        return self._df_balance

    def plot_balance(self, logy=False, title=None):
        """Show plot of account balance over time"""
        # fig, axs = plt.subplots(figsize=(12, 6))
        self.df_balance.plot(kind='line', y='balance', logy=logy, linewidth=1, title=title, figsize=(12, 4))

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
        df = pd.DataFrame(columns=['Timestamp', 'Balance', 'PercentChange'])
        for i, t in enumerate(self.txns):
            df.loc[i] = [t.timestamp, t.balance_pre, t.percentchange]
        return df
