from .__init__ import *
from .base import Observer, SignalEvent
from .orders import LimitOrder, MarketOrder, Order, OrderType, StopOrder
from .wallet import Wallet

log = getlog(__name__)


class Broker(Observer):
    """Class to manage submitting and checking orders
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        all_orders = {}
        filled_orders = {}
        open_orders = {}
        wallets = {}

        # temp set default wallet to only XBTUSD
        symbol = 'XBTUSD'
        wallets[symbol] = Wallet(symbol=symbol)

        f.set_self(vars())

    def get_wallet(self, symbol: str):
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

    def submit(self, orders: Union[list, 'Order']):
        """Submit single or multiple orders at once

        Parameters
        ----------
        orders : Union[list,
            order(s) to submit
        """

        if not isinstance(orders, list):
            orders = [orders]

        for order in orders:
            self._submit_single(order=order)

        # sort after all alled
        self.open_orders = {order.order_id: order for order in sorted(
            self.open_orders.values(), key=lambda x: x.sort_key)}

    def _submit_single(self, order: 'Order'):
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
            self.open_orders[order.order_id] = order

    def cancel(self, order: 'Order'):
        """Cancel order"""
        self.open_orders.pop(order.order_id)
        order.cancel()

    def amend(self, order: 'Order', price: float = None, qty: int = None):
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

        order.ammended.emit()

    def fill_order(self, order: 'Order'):
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
                order.price = self.c.Close

            # fill if order lies in current candle's range
            if self.c.Low <= order.price <= self.c.High:
                self.fill_order(order)

            # TODO handle wallet transactions here

            # TODO check expired (timedout) orders
            if order.is_expired:
                order.cancel()
                # NOTE this should remove from open orders and clear all listeners?
