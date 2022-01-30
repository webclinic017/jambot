import math

from jambot.tradesys import orders as ords
from jambot.tradesys.wallet import Wallet

from .__init__ import *


@fixture
def wallet(clock):
    wallet = Wallet(symbol=SYMBOL, parent_listener=clock)
    clock.next()
    return wallet


def test_market_orders(wallet: Wallet):
    """Test simple buying and selling in sequence"""

    o1 = ords.make_order(order_type='market', price=100, qty=100)
    o2 = ords.make_order(order_type='market', price=120, qty=100)
    o3 = ords.make_order(order_type='market', price=140, qty=-200)

    wallet.fill_order(o1)

    assert wallet.price == 100

    wallet.fill_order(o2)

    assert wallet.price == 110
    assert wallet.qty == 200

    wallet.fill_order(o3)

    assert wallet.qty == 0
    assert wallet.price == 0
    assert math.isclose(wallet.balance, 1.38716, rel_tol=0.001), f'wallet balance is: {wallet.balance:.5f}'
    assert wallet.num_txns == 1


def test_limit_locking(wallet):

    pass
