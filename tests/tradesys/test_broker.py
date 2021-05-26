from jambot.tradesys import orders as ords
from jambot.tradesys.broker import Broker

from .__init__ import *

# from .test_orders import orders

# @mark.usefixtures('order')


def test_boker(orders):
    """Test broker submits and sorts orders correctly"""
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
