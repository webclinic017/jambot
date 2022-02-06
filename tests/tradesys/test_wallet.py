import math

from pytest import fixture

from jambot.tradesys import orders as ords
from jambot.tradesys.symbols import Symbol
from jambot.tradesys.wallet import Wallet


@fixture
def wallet(clock, symbol: Symbol):
    wallet = Wallet(symbol=symbol, parent_listener=clock)

    # linear contracts need higher start balance
    if not symbol.is_inverse:
        wallet.balance = 100000

    clock.next()
    return wallet


def test_market_orders(wallet: Wallet, symbol: Symbol):
    """Test simple buying and selling in sequence"""

    o1 = ords.make_order(order_type='market', price=100, qty=100, symbol=symbol)
    o2 = ords.make_order(order_type='market', price=120, qty=100, symbol=symbol)
    o3 = ords.make_order(order_type='market', price=140, qty=-200, symbol=symbol)

    wallet.fill_order(o1)

    assert wallet.price == 100

    wallet.fill_order(o2)

    assert wallet.price == 110
    assert wallet.qty == 200

    wallet.fill_order(o3)

    assert wallet.qty == 0
    assert wallet.price == 0

    # kinda ugly, inverse and linear perform differently though
    final_bal = 1.38716 if symbol.is_inverse else 105975.0

    assert math.isclose(wallet.balance, final_bal, rel_tol=0.001), f'wallet balance is: {wallet.balance:.5f}'
    assert wallet.num_txns == 1


def test_limit_locking(wallet):

    pass
