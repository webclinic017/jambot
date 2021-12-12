from datetime import datetime as dt
from typing import *
from typing import Union

import numpy as np
import pandas as pd
from jgutils import functions as jf

from jambot import SYMBOL, display
from jambot import functions as f
from jambot import getlog
from jambot.exchanges.exchange import SwaggerExchange
from jambot.tradesys.base import Observer
from jambot.tradesys.enums import OrderStatus
from jambot.tradesys.orders import ExchOrder, Order
from jambot.tradesys.wallet import Wallet

log = getlog(__name__)


class Broker(Observer):
    """Class to manage submitting and checking orders
    """

    def __init__(self, symbol: str = SYMBOL, exch_name: str = 'bitmex', *args, **kw):
        super().__init__(*args, **kw)
        all_orders = {}
        open_orders = {}
        wallets = {}

        # temp set default wallet to only XBTUSD
        symbol = symbol.lower()
        wallets[symbol] = Wallet(symbol=symbol, exch_name=exch_name)
        self.attach_listeners(wallets.values())

        jf.set_self()

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
        return self.wallets[symbol.lower()]

    def submit(self, orders: Union[list, Order]):
        """Submit single or multiple orders at once

        Parameters
        ----------
        orders : Union[list,
            order(s) to submit
        """

        for order in jf.as_list(orders):
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
            order.price = self.c.close
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
        wallet = self.get_wallet(order.symbol)
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
                order.price = self.c.close

            # fill if order lies in current candle's range
            # NOTE technically this should just check if price > low or < high, per side
            if self.c.low <= order.price <= self.c.high:

                self.fill_order(order)

            elif order.is_expired:
                self.cancel_order(order)

                # NOTE not sure if broker should be the one to do this
                order.timedout.emit(order)

    def recent_markets(self, ts: dt = None) -> List[Order]:
        """Get list of market orders filled during current timestamp

        Parameters
        ----------
        ts : dt, optional
            timestamp to check, by default last timestamp in df

        Returns
        -------
        List[Order]
            list of market orders filled
        """
        # NOTE might need to be previous timestamp
        if ts is None:
            ts = self.timestamp

        return [o for o in self.all_orders.values() if o.is_market and o.timestamp_filled == ts]

    def _open_orders(self) -> List[Order]:
        """Get list of all open orders (wrapper for self.open_orders dict)

        Returns
        -------
        List[Order]
            list of all open orders
        """
        return list(self.open_orders.values())

    def expected_orders(self, symbol: str, exch: SwaggerExchange = None) -> List[ExchOrder]:
        """Get all market/limit orders to check for current timestamp
        - Must be called with user-specific Exchange
        - NOTE this will currently just scale orders based on max avail qtys
        - May need to support multiple orders in the future (eg close half of position)

        Returns
        -------
        List[Order]
            list of all orders
        """
        orders = self.recent_markets() + self._open_orders()
        orders = [o.as_exch_order() for o in orders]

        # rescale orders
        # TODO stops need to be related to limit open
        if not exch is None:
            wallet = self.get_wallet(symbol)
            wallet.set_exchange_data(exch)  # IMPORTANT
            expected_side = wallet.side
            cur_qty = exch.current_qty(symbol=symbol)
            cur_side = np.sign(cur_qty)
            # last_price = exch.last_price(symbol=symbol)

            for o in orders:
                # if o.is_limit:
                #     # order is offside (price moved too fast from close)
                #     # adjust close price to exch's last price + offset %
                #     if (o.price - last_price) * o.side > 0:
                #         o.price = f.get_price(pnl=o.offset, price=last_price, side=o.side)

                if o.is_reduce:
                    if not o.is_stop:
                        o.qty = cur_qty * -1
                    else:
                        raise NotImplementedError('Stop order rescaling not set up yet.')

                elif o.is_increase:
                    o.qty = wallet.available_quantity(price=o.price) * o.side

            # consider impact of submitting market orders this period first
            cur_qty += sum([o.qty for o in orders if o.is_market])
            cur_side = np.sign(cur_qty)
            # print('cur_side: ', cur_side)

            # final checks to get position back to correct side
            if not expected_side == cur_side:
                # print('expected side not equal')

                # Market Close
                if expected_side * cur_side == -1:
                    log.warning(f'Position offside, market closing. Expected: {expected_side}, actual: {cur_side}')

                    # remove limit close
                    orders = [o for o in orders if not o.is_limit_close]

                    # add market close
                    ExchOrder.market(
                        exch_name=exch.exch_name,
                        symbol=symbol,
                        qty=cur_qty * -1,
                        name='mkt_close_er').add(orders)

                # Trailing limit open
                has_opens = len([o for o in orders if o.is_increase]) > 0
                if not has_opens:

                    limit_price = f.get_price(
                        pnl=self.parent.order_offset,
                        price=exch.last_price(symbol=symbol),
                        side=expected_side)

                    # # NOTE could enforce this as Limit only, keep retrying till success
                    lim_er = ExchOrder.limit(
                        exch_name=exch.exch_name,
                        symbol=symbol,
                        price=limit_price,
                        qty=wallet.available_quantity(price=limit_price) * expected_side,
                        name='lim_open_er',
                        prevent_market_fill=False).add(orders)

                    log.warning(f'Adding limit_open_er: {lim_er}')

        return orders

    @property
    def df_orders(self) -> pd.DataFrame:
        data = [o.to_dict() for o in self.all_orders.values()]
        return pd.DataFrame.from_dict(data)

    def show_orders(self, last: int = 30) -> None:
        display(self.df_orders.iloc[-last:])
