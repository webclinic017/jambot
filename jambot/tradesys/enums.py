# from enum import Enum, IntEnum
from aenum import IntEnum, MultiValueEnum


class CaseInsensitiveEnum(MultiValueEnum):
    """Base enum to allow initialization with any case string"""

    @classmethod
    def _missing_(cls, name):
        """Find matching member when case insensitve or multiple values"""
        for member in cls:
            if any(val.lower() == str(name).lower() for val in member.values):
                return member

    def __str__(self):
        return str(self.value)


class OrderType(CaseInsensitiveEnum):
    """Enum for type of order"""
    LIMIT: str = 'limit'
    MARKET: str = 'market'
    STOP: str = 'stop'


class OrderStatus(CaseInsensitiveEnum):
    """Enum for the status of an order"""

    PENDING: str = 'pending'
    OPEN: str = 'open', 'new'
    CANCELLED: str = 'cancelled', 'canceled'
    FILLED: str = 'filled'
    PARTIALLYFILLED: str = 'partiallyfilled'
    REJECTED: str = 'rejected'


class TradeStatus(CaseInsensitiveEnum):
    """Enum for trade status"""
    PENDING: str = 'pending'  # no orders filled yet
    OPEN: str = 'open'  # first order filled
    CLOSED: str = 'closed'  # all orders filled


class TradeSide(IntEnum):
    """Enum for side of trade"""
    LONG = 1
    NEUTRAL = 0
    SHORT = -1

    def __str__(self):
        return str(self.value)
