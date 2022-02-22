from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from jambot.tradesys.orders import Order


class BaseError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InsufficientBalance(BaseError):
    """
    Raised when order attempted to fill with insufficient balance available
    """

    def __init__(
            self,
            balance: float,
            used_qty: int,
            order_qty: int,
            avail_qty: int,
            order: object = None,
            *args) -> None:

        msg = f'Insufficient balance [{balance:.3f}] ' + \
            f'for order of quantity [{order_qty:,.0f}]. ' + \
            f'Used: [{used_qty:,.0f}], Available: [{avail_qty:,.0f}]'

        if not order is None:
            msg = f'{msg}\n{order}'

        super().__init__(msg, *args)


class PositionNotClosedError(BaseError):
    """
    Raised when exchange position for specific symbol should have zero qty but does not
    """

    def __init__(self, qty: int, *args) -> None:
        msg = f'Position not closed. Expected: 0, Actual: {qty}'
        super().__init__(msg, *args)


class InvalidTradeOperationError(BaseError):
    """
    Raised when trade attempted to be closed while contracts still open
    """

    def __init__(self, qty_open: int, trade_num: int, orders: List['Order'], *args) -> None:
        _orders = '\n'.join([str(o) for o in orders])
        msg = f'Cant close trade [{trade_num}] with [{qty_open}] contracts open!\n\n{_orders}'
        super().__init__(msg, *args)
