from IPython.display import display

from .__init__ import *
from .base import Observer, SignalEvent
from .enums import OrderStatus
from .orders import LimitOrder, MarketOrder, Order, OrderType, StopOrder
from .wallet import Wallet

log = getlog(__name__)


class Broker(Observer):
    """Class to manage submitting and checking orders
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        all_orders = {}
        filled_orders = {}
        open_orders = {}
        wallets = {}

        # temp set default wallet to only XBTUSD
        symbol = 'XBTUSD'
        wallets[symbol] = Wallet(symbol=symbol)
        self.attach_listeners(wallets.values())

        f.set_self(vars())

    def get_wallet(self, symbol: str) -> Wallet:
        """Get wallet for specific trade pair (symbol)

        Parameters
        ----------
        symbol : str
            trade pair (XBTUSD)

        Returns
        -------
        Wallet
        """
        return self.wallets[symbol]

    def submit(self, orders: Union[list, Order]):
        """Submit single or multiple orders at once

        Parameters
        ----------
        orders : Union[list,
            order(s) to submit
        """

        for order in f.as_list(orders):
            self._submit_single(order=order)

        # sort after all added
        self.open_orders = {order.order_id: order for order in sorted(
            self.open_orders.values(), key=lambda x: x.sort_key)}

    def _submit_single(self, order: Order):
        """Add order to open orders, fill market immediately

        Parameters
        ----------
        order : Order
            [description]
        """
        self.all_orders[order.order_id] = order

        # TODO need to mock filling?
        if order.is_market and not self.c is None:
            # fill immediately
            order.price = self.c.Close
            self.fill_order(order)
        else:
            order.status = OrderStatus.OPEN
            self.open_orders[order.order_id] = order

    def cancel_order(self, order: Order) -> None:
        """Cancel order"""
        if order.order_id in self.open_orders:
            self.open_orders.pop(order.order_id)
            order.cancel()

    def amend_order(self, order: Order, price: float = None, qty: int = None) -> None:
        """Change order price or quantity

        Parameters
        ----------
        order : Order
        price : float
            new order price
        qty : int
            new order quantity
        """
        if not price is None:
            order.price = price

        if not qty is None:
            order.qty = qty

        order.amended.emit()

    def fill_order(self, order: Order) -> None:
        """Execute order at specific price"""
        wallet = self.wallets[order.symbol]
        wallet.fill_order(order)

        # remove order from open orders
        if order.order_id in self.open_orders:
            self.open_orders.pop(order.order_id)

    def step(self):
        """Check open orders, fill if price hit
        - NOTE should probably consider order candle hits either side
        """

        for order_id in list(self.open_orders):
            # need order direction, check relative to price
            order = self.open_orders[order_id]

            if order.is_market:
                # NOTE cant adjust market order price without adjusting price
                order.price = self.c.Close

            # fill if order lies in current candle's range
            if self.c.Low <= order.price <= self.c.High:

                self.fill_order(order)

            elif order.is_expired:
                self.cancel_order(order)

                # NOTE not sure if broker should be the one to do this
                order.timedout.emit(order)

    @property
    def df_orders(self) -> pd.DataFrame:
        data = [o.to_dict() for o in self.all_orders.values()]
        return pd.DataFrame.from_dict(data)

    def show_orders(self, last: int = 30) -> None:
        display(self.df_orders.iloc[-last:])
