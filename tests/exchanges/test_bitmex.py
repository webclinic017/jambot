import time
from typing import List

import pytest
from pytest import fixture, mark, raises

from jambot import functions as f
from jambot import getlog
from jambot.exchanges.exchange import SwaggerAPIException, SwaggerExchange
from jambot.livetrading import ExchangeManager
from jambot.tradesys import orders as ords
from jambot.tradesys.enums import OrderStatus
from jambot.tradesys.orders import ExchOrder
from jambot.tradesys.symbols import Symbol, Symbols
from tests import mul

log = getlog(__name__)


@fixture(scope='session')
def exch(exch_name: str, em: ExchangeManager) -> SwaggerExchange:
    """Exchange object"""
    return em.default(exch_name=exch_name, test=True, refresh=True)


@fixture(scope='session')
def syms(em: ExchangeManager) -> Symbols:
    """Symbols manager, init once with ExchangeManager first"""
    return em.syms


@fixture(scope='session')
def symbol(exch) -> str:
    return exch.default_symbol


@fixture(scope='session')
def exch_name(exch) -> str:
    return exch.exch_name


def test_exch_is_test(exch) -> None:
    """Make sure exch is in test mode"""
    assert exch.test is True


@fixture(scope='session')
def last_close(exch: SwaggerExchange, symbol: Symbol) -> float:
    """Get last close price offset by 15% higher to use as base for creating new orders"""
    price = f.get_price(pnl=-0.15, price=exch.last_price(symbol=symbol), side=-1)
    return round(price, 0)


@fixture
def exch_orders(last_close: float, symbol: Symbol, exch_name: str, syms: Symbols) -> List[ExchOrder]:
    """Create two bitmex orders for testing"""
    order_specs = [
        dict(symbol=symbol, order_type='limit', qty=mul(-1, symbol), price=last_close, name='test_ord_1'),
        dict(symbol=symbol, order_type='limit', qty=mul(-1, symbol),
             price=last_close + mul(1, symbol), name='test_ord_2'),
    ]

    return ords.make_exch_orders(order_specs, exch_name=exch_name, syms=syms)


def test_order_flow(exch: SwaggerExchange, exch_orders: List[ExchOrder], symbol: Symbol):
    """Test submitting, amending, cancelling and comparing in/out order specs for multiple orders"""
    exch.cancel_all_orders(symbol=symbol)

    # submit orders
    out_orders = exch.submit_orders(exch_orders)

    try:
        # compare specs
        for order_in, order_out in zip(exch_orders, out_orders):
            _compare_order_specs(order_in, order_out)

            # amend order
            order_out.price += mul(1, symbol)
            order_out.increase_qty(mul(1, symbol))

        amend_orders = exch.amend_orders(out_orders)
        assert all(order_amend.qty == mul(-2, symbol) for order_amend in amend_orders)

    except Exception as e:
        pytest.fail(str(e))

    finally:
        # cancel order
        cancelled_orders = exch.cancel_orders(out_orders)
        assert all(order_cancelled.is_cancelled for order_cancelled in cancelled_orders), 'Order not cancelled.'


@mark.skip
def _compare_order_specs(order_1: ExchOrder, order_2: ExchOrder) -> None:
    """Test order specs match

    Parameters
    ----------
    order_1 : ExchOrder
    order_2 : ExchOrder

    Raises
    ------
    AssertionError
        if order specs don't match
    """
    compare = ['symbol', 'order_type', 'key', 'qty']

    # market orders don't have price to compare if not filled
    if not order_1.is_market:
        compare.append('price')

    spec_in = order_1.order_spec
    spec_out = order_2.order_spec

    for k in compare:
        item_in, item_out = spec_in.get(k, None), spec_out.get(k, None)
        assert item_in == item_out, \
            f'Order specs don\'t match: \
            order[{k}]={item_in}, order_out[{k}]={item_out}\n\n{order_1}\n{order_2}'


def test_reconcile_orders(
        exch: SwaggerExchange,
        last_close: float,
        symbol: Symbol,
        syms: Symbols) -> None:
    """Test amending/cancelling/submitting orders from strategy
    - TODO test position offside > add lim_open_er
    - TODO test wallets have been refreshed to set total_balance_margin

    - create matched, missing, and not_matched orders for limit/stop/market
    """
    if not exch.current_qty(symbol=symbol) == 0:
        exch.close_position(symbol=symbol)

    exch.cancel_all_orders(symbol=symbol)
    # TODO test current_qty somewhere big oops made it into prod!!
    # NOTE could set qtys/price dynamic per symbol by checking exch max qty but too lazy

    try:
        # submit test orders
        order_specs_test = [
            dict(symbol=symbol, order_type='limit', qty=mul(-12.34, symbol), price=last_close, name='limit_1'),  # amend
            dict(symbol=symbol, order_type='limit', qty=mul(-22, symbol),
                 price=last_close + mul(4, symbol), name='limit_2'),  # cancel
        ]

        test_orders = exch.submit_orders(order_specs_test)

        # set 'expected' orders (from strat)
        order_specs_expected = [
            dict(symbol=symbol, order_type='limit', qty=mul(-15, symbol),
                 price=last_close + mul(-2, symbol), name='limit_1'),  # amend
            dict(symbol=symbol, order_type='stop', qty=mul(12.34, symbol),
                 price=last_close + mul(5, symbol), name='stop_1'),  # submit
            dict(symbol=symbol, order_type='market', qty=mul(-1, symbol), name='market_3'),  # submit
        ]

        expected_orders = ords.make_exch_orders(order_specs_expected, exch_name=exch.exch_name, syms=syms)

        # reconcile - cancel, amend, submit
        exch.reconcile_orders(symbol=symbol, expected_orders=expected_orders, bybit_async=True, bybit_stops=True)

        # assert correct orders submitted, cancelled, and amended
        if exch.exch_name == 'bybit':
            # bybit api seems to be too slow even with async orders (only sometimes)
            log.warning('Sleeping 20s for Bybit')
            time.sleep(20)

        final_orders = exch.get_orders(
            symbol=symbol,
            bot_only=True,
            new_only=False,
            as_exch_order=True,
            as_dict=True,
            refresh=True,
            bybit_async=True,
            bybit_stops=True)

        # need to reference existing order's key to check amended
        expected_orders[0].key = test_orders[0].key
        expected_orders = ords.list_to_dict(expected_orders, use_ts=True)

        for order_key, o in expected_orders.items():
            assert order_key in list(final_orders.keys()), f'Failed to find order: {order_key}'

            o_actual = final_orders[order_key]
            _compare_order_specs(order_1=o, order_2=o_actual)

            if o_actual.is_market:
                assert o_actual.status == OrderStatus.FILLED, f'Order: {o_actual.key} not filled!'
            else:
                assert o_actual.status == OrderStatus.OPEN, f'Order: {o_actual.key} not open!'

        order_cancel = final_orders.get(test_orders[1].key)
        assert order_cancel.status == OrderStatus.CANCELLED

    finally:
        # cancel everything
        exch.close_position(symbol=symbol)
        exch.cancel_all_orders(symbol=symbol)


@mark.skip
def test_cancel_warning(exch, last_close):
    """Test cancel warning sent if bad order sent"""
    # order_in = LimitOrder(price=last_price - 10000, qty=-100, name='bad_order').as_exch_order()

    # order_out = exch.submit_orders(order_in)
    return


def test_api_exception(exch: SwaggerExchange, symbol: Symbol, syms: Symbols):
    """Test correct exception raised with invalid data to exchange"""

    # orderID length too short
    with raises(SwaggerAPIException):
        order_spec = {exch.order_keys['order_id']: '1234', 'symbol': symbol}
        # exch.req('Order.cancel', **kw)
        exch.cancel_orders(orders=order_spec)

    # invalid price
    with raises(SwaggerAPIException):
        spec = dict(order_type='limit', symbol=symbol, price=-5, qty=mul(1, symbol), name='test_excep')
        orders = ords.make_exch_orders(order_specs=spec, exch_name=exch.exch_name, syms=syms)
        exch.submit_orders(orders)
