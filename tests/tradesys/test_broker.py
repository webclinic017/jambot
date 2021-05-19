from pytest import raises

from jambot.tradesys import orders as ords
from jambot.tradesys.broker import Broker


def test_boker():
    """Test broker submits and sorts orders correctly"""
    order_specs = [
        dict(order_type='limit', qty=-5000, price=10000),
        dict(order_type='limit', qty=5000, price=9000),
        dict(order_type='stop', qty=-5000, price=8900),
        dict(order_type='stop', qty=5000, price=10100),
        dict(order_type='market', qty=-11111),
        dict(order_type='market', qty=11111),
    ]

    orders = ords.make_orders(order_specs)

    broker = Broker()
    broker.submit(orders)

    expected = [
        ('market', None),
        ('market', None),
        ('limit', 9000),
        ('stop', 8900),
        ('limit', 10000),
        ('stop', 10100),
    ]

    for (order_type, price), order in zip(expected, broker.open_orders.values()):
        assert order_type == str(order.order_type)
        assert price == order.price
