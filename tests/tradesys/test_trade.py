from jambot import data
from jambot.tradesys import orders as ords
from jambot.tradesys.base import Clock
from jambot.tradesys.broker import Broker
from jambot.tradesys.enums import TradeSide, TradeStatus
from jambot.tradesys.trade import Trade

from .__init__ import *


@fixture
def broker():
    return Broker()


@fixture
def trade(broker):
    return Trade(symbol=SYMBOL, broker=broker)


@fixture
def clock():
    df = data.default_df()
    clock = Clock(df=df)
    # clock.attach(broker)
    return clock


def test_init(trade):
    assert trade.symbol == SYMBOL


def test_orders(trade, clock):
    """Test orders are added and removed correctly when filled"""
    clock.attach(trade)
    clock.attach(trade.broker)

    order_specs = [
        dict(price=100, qty=1000, name='o1'),
        dict(price=120, qty=1000, name='o2')
    ]

    orders = ords.make_orders(order_specs, order_type='market', symbol=SYMBOL)
    o1, o2 = orders[0], orders[1]

    trade.add_orders(orders)

    assert trade.num_listeners == 2

    clock.next()

    assert trade.num_listeners == 0

    assert clock.duration == 1
    assert trade.duration == 1
    assert o1.duration == 1
    assert trade.broker.duration == 1

    # orders filled after first clock step
    assert trade.is_open
    assert trade.side == TradeSide.LONG
    assert trade.wallet.qty == 2000

    clock.next()
    trade.market_close()

    assert trade.is_closed
