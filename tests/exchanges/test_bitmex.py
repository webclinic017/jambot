from jambot.exchanges.bitmex import Bitmex
from jambot.tradesys import orders as ords
from jambot.tradesys.orders import LimitOrder

from .__init__ import *


@fixture(scope='session')
def exch():
    """Bitmex exchange object"""
    return Bitmex.default(test=True, refresh=True)


def test_exch_is_test(exch):
    """Make sure exch is in test mode"""
    assert exch.test is True


@fixture(scope='session')
def last_close(exch):
    """Get last close price to use as base for creating new orders"""
    price = exch.get_position(SYMBOL)['prevClosePrice']
    price = f.get_price(pnl=-0.2, entry_price=price, side=-1)
    return round(price, 0)


@fixture
def bitmex_orders(last_close):
    """Create two bitmex orders for testing"""
    order_specs = [
        dict(order_type='limit', qty=-100, price=last_close, name='test_ord_1'),
        dict(order_type='limit', qty=-100, price=last_close + 100, name='test_ord_2'),
    ]

    return ords.make_orders(order_specs, as_bitmex=True)


def test_order_flow(exch, bitmex_orders):
    """Test submitting, amending, cancelling and comparing in/out order specs for multiple orders"""

    # submit orders
    out_orders = exch.submit_orders(bitmex_orders)

    try:
        # compare specs
        compare = ['symbol', 'ordType', 'clOrdID', 'price', 'orderQty']

        for order_in, order_out in zip(bitmex_orders, out_orders):
            spec_in = order_in.order_spec
            spec_out = order_out.order_spec

            for k in compare:
                assert spec_in[k] == spec_out[k], \
                    f'Order specs don\'t match: \
                    order[{k}]={spec_in[k]}, order_out[{k}]={spec_out[k]}'

            # amend order
            order_out.price += 100
            order_out.increase_qty(10)

        amend_orders = exch.amend_orders(out_orders)
        assert all(order_amend.qty == -110 for order_amend in amend_orders)

    except Exception as e:
        pytest.fail(str(e))

    finally:
        # cancel order
        cancelled_orders = exch.cancel_orders(out_orders)
        assert all(order_cancelled.is_cancelled for order_cancelled in cancelled_orders), 'Order not cancelled.'


def test_cancel_warning(exch, last_price):
    """Test cancel warning sent if bad order sent"""
    # order_in = LimitOrder(price=last_price - 10000, qty=-100, name='bad_order').as_bitmex()

    # order_out = exch.submit_orders(order_in)
    return
