from jambot.tradesys import orders as ords
from jambot.tradesys.orders import BitmexOrder

from .__init__ import *


@fixture(scope='session')
def orders():
    """Create list of orders"""
    order_specs = [
        dict(order_type='limit', qty=-5000, price=10000),
        dict(order_type='limit', qty=5000, price=9000),
        dict(order_type='stop', qty=-5000, price=8900),
        dict(order_type='stop', qty=5000, price=10100),
        dict(order_type='market', qty=-11111),
        dict(order_type='market', qty=11111),
    ]

    orders = ords.make_orders(order_specs)
    return orders


@fixture(scope='session')
def order(orders):
    """Create single order"""
    return orders[0]


def test_orders(orders):
    assert len(orders) == 6


def test_bitmex_order(order):
    """Test backtesting orders can be converted to BitmexOrder"""
    bitmex_order = BitmexOrder.from_base_order(order)
    s = bitmex_order.to_json()


def test_make_bitmex_orders(exch):
    """Test order spec dicts can be converted to BitmexOrders"""
    order_specs = exch.orders[0:3]
    bitmex_orders = ords.make_bitmex_orders(order_specs)
