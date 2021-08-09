
class BaseError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InsufficientBalance(BaseError):
    """Raised when order attempted to fill with insufficient balance available
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
