from typing import *

from jambot.tradesys import orders as ords
from jambot.tradesys.broker import Broker
from jambot.tradesys.enums import TradeSide
from jambot.tradesys.trade import Trade

from .__init__ import *

if TYPE_CHECKING:
    from jambot.tradesys.symbols import Symbol


@fixture
def broker(clock):
    return Broker(parent_listener=clock)


@fixture
def trade(broker, clock, symbol: 'Symbol' = SYMBOL):
    return Trade(symbol=symbol, broker=broker, parent_listener=clock)


def test_init(trade, symbol: 'Symbol' = SYMBOL):
    assert trade.symbol == symbol


def test_orders(trade: Trade, clock: Clock, symbol: 'Symbol' = SYMBOL):
    """Test orders are added and removed correctly when filled"""

    order_specs = [
        dict(price=100, qty=1000, name='o1'),
        dict(price=120, qty=1000, name='o2')
    ]

    orders = ords.make_orders(order_specs, order_type='market', symbol=symbol)
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
    assert trade.wallet.qty == 0
