from typing import *

from jambot.tradesys import orders as ords
from jambot.tradesys.orders import ExchOrder, Order

from .__init__ import *

if TYPE_CHECKING:
    from jambot.exchanges.exchange import SwaggerExchange
    from jambot.tradesys.symbols import Symbols


@fixture(scope='session')
def orders() -> List[Order]:
    """Create list of orders
    - NOTE Order object will be init with default Symbol(XBTUSD)
    - not sure how these prices/qtys would need to be adjusted to test ALL symbols
    """
    order_specs = [
        dict(order_type='limit', qty=-5000, price=10000),
        dict(order_type='limit', qty=5000, price=9000),
        dict(order_type='stop', qty=-5000, price=8900),
        dict(order_type='stop', qty=5000, price=10100),
        dict(order_type='market', qty=-11111),
        dict(order_type='market', qty=11111),
    ]

    orders = ords.make_orders(order_specs, as_exch_order=False)  # type: List[Order]
    return orders


@fixture(scope='session')
def order(orders) -> Order:
    """Create single order"""
    return orders[0]


def test_orders(orders):
    assert len(orders) == 6


def test_bitmex_order(order):
    """Test backtesting orders can be converted to ExchOrder"""
    bitmex_order = ExchOrder.from_base_order(order)
    s = bitmex_order.to_json()


def test_make_bitmex_orders(exch: 'SwaggerExchange', syms: 'Symbols'):
    """Test order spec dicts can be converted to ExchOrders"""
    order_specs = exch.orders[0:3]
    exch_orders = ords.make_exch_orders(order_specs, exch_name=exch.exch_name, syms=syms)
