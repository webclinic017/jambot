import math

from jambot.tradesys import orders as ords
from jambot.tradesys.wallet import Wallet

symbol = 'XBTUSD'


def test_init():
    wallet = Wallet(symbol=symbol)


def test_fill_order():
    """Test simple buying and selling in sequence"""
    wallet = Wallet(symbol=symbol)

    o1 = ords.make_order(order_type='market', price=100, qty=1000)
    o2 = ords.make_order(order_type='market', price=120, qty=1000)
    o3 = ords.make_order(order_type='market', price=140, qty=-2000)

    wallet.fill_order(o1)

    assert wallet.price == 100

    wallet.fill_order(o2)

    assert wallet.price == 110
    assert wallet.qty == 2000

    wallet.fill_order(o3)

    assert wallet.qty == 0
    assert wallet.price == 0
    assert math.isclose(wallet.balance, 4.8961, rel_tol=0.001)
    assert wallet.num_txns == 1
