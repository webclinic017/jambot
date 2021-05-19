from enum import Enum, IntEnum


class OrderType(Enum):
    """Enum for type of order (better readability)"""
    LIMIT: str = 'limit'
    MARKET: str = 'market'
    STOP: str = 'stop'

    def __str__(self):
        return str(self.value)


class OrderStatus(Enum):
    """An enumeration for the status of an order."""

    PENDING: str = 'pending'
    OPEN: str = 'open'
    CANCELLED: str = 'cancelled'
    FILLED: str = 'filled'

    def __str__(self):
        return str(self.value)


class TradeSide(IntEnum):
    """Enum for side of trade"""
    LONG = 1
    NEUTRAL = 0
    SHORT = -1

    def __str__(self):
        return str(self.value)


class TradeStatus(Enum):
    """Enum for trade status"""
    PENDING: str = 'pending'  # no orders filled yet
    OPEN: str = 'open'  # first order filled
    CLOSED: str = 'closed'  # all orders filled

    def __str__(self):
        return str(self.value)
