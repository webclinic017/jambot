"""General Functions module - don't rely on any other modules from jambot"""

import math
from datetime import date
from datetime import datetime as dt
from datetime import timedelta as delta
from typing import TYPE_CHECKING, Any, List, Union

import pandas as pd

from jambot import getlog

if TYPE_CHECKING:
    from pandas._libs.missing import NAType

log = getlog(__name__)


def flatten_list_list(lst: List[list]) -> list:
    """Flatten single level nested list of lists

    Parameters
    ----------
    lst : List[list]

    Returns
    -------
    list
        flattened list
    """
    return [item for sublist in lst for item in sublist]


def str_to_num(val: Any) -> Union[Any, float, int]:
    """Convert string float/int to correct type

    Parameters
    ----------
    val : Any
        any value to try

    Returns
    -------
    Union[Any, float, int]
        int | float | Any (original value)
    """
    if isinstance(val, (float, int)):
        return val

    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def percent(x: float) -> Union[str, 'NAType']:
    """Format float as string percent"""
    return f'{x:.2%}' if pd.notna(x) else pd.NA


def round_down(n: Union[int, float], nearest: Union[int, float] = 100) -> float:
    """Round down to nearest value (used for round contract sizes to nearest 100)

    Parameters
    ----------
    n : Union[int, float]
        number to round
    nearest : int, optional
        value to round to, by default 100

    Returns
    -------
    int
        number rounded to nearest
    """
    return int(math.floor(n / nearest)) * nearest


def get_price(pnl: float, price: float, side: int, prec: float = 0.5) -> float:
    """Get price at percentage offset

    Parameters
    ----------
    pnl : float
        percent offset from entry_price
    price : float
        price to offset from
    side : int
        order side

    Returns
    -------
    float
        price
    """
    if side == 1:
        price = pnl * price + price
    elif side == -1:
        price = price / (1 + pnl)

    return round_down(n=price, nearest=prec)


def get_pnl_xbt(qty: int, entry_price: float, exit_price: float, isaltcoin: bool = False) -> float:
    """Get profit/loss in base currency (xbt) from entry/exit

    Parameters
    ----------
    qty : int, optional
        quantity of contracts
    entry_price : float, optional
    exit_price : float, optional
    isaltcoin : bool, optional
        alts calculated opposite, default False
        - NOTE this should all be restructured to be relative to a base currency somehow

    Returns
    -------
    float
        amount of xbt gained/lossed
    """
    if 0 in (entry_price, exit_price):
        return 0.0
    elif not isaltcoin:
        return round(qty * (1 / entry_price - 1 / exit_price), 8)
    else:
        # is altcoin
        return round(qty * (exit_price - entry_price), 8)


def get_pnl(side, entry_price, exit_price):
    if 0 in (entry_price, exit_price):
        return 0
    elif side == 1:
        return round((exit_price - entry_price) / entry_price, 4)
    else:
        return round((entry_price - exit_price) / exit_price, 4)


def get_contracts(xbt, leverage, price, side, is_altcoin=False):
    if not is_altcoin:
        return int(xbt * leverage * price * side)
    else:
        return int(xbt * leverage * (1 / price) * side)


def side(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def useful_keys(orders):
    """return only useful keys from bitmex orders in dict form
    - NOTE don't think this is needed anymore
    """
    keys = ('symbol', 'clOrdID', 'side', 'price', 'stopPx', 'ordType',
            'execInst', 'ordStatus', 'qty', 'name', 'manual', 'orderID')

    if not isinstance(orders, list):
        islist = False
        orders = [orders]
    else:
        islist = True

    result = [{key: o[key] for key in o.keys() if key in keys} for o in orders]

    if islist:
        return result
    else:
        return result[0]


def col(df, col):
    return df.columns.get_loc(col)


def inter_offset(interval: int = 1) -> delta:
    """Get single period offset for 1hr or 15m

    Parameters
    ----------
    interval : int, optional
        default 1 hr

    Returns
    -------
    datetime.timedelta
        offset for selected interval
    """
    return {1: delta(hours=1), 15: delta(minutes=15)}[interval]


def date_to_dt(d: date) -> dt:
    return dt.combine(d, dt.min.time())


def inter_now(interval: int = 1) -> dt:
    """Get current utc time rounded down to nearest 1hr/15min interval

    Parameters
    ----------
    interval : int, optional
        default 1

    Returns
    -------
    dt
        datetime rounded to interval
    """
    if interval == 1:
        return dt.utcnow().replace(microsecond=0, second=0, minute=0)
    elif interval == 15:
        return round_minutes(dt=dt.utcnow(), resolution=15).replace(microsecond=0, second=0)
    else:
        raise ValueError('Incorrect interval.')


def round_minutes(dt, resolution):
    new_minute = (dt.minute // resolution) * resolution
    return dt + delta(minutes=new_minute - dt.minute)


def ci_rate(final: float, periods: int, initial: float = 1.0) -> float:
    """Calc compount interest rate
    - r = (A / P)^(1 / n) - 1

    Parameters
    ----------
    final : float
        final value
    periods : int
        num compounding periods
    initial : float, optional
        initial value, default 1.0

    Returns
    -------
    float
        interest rate
    """
    return (final / initial) ** (1 / periods) - 1


def ci_final(periods: int, rate: float, initial: float = 1.0) -> float:
    """Calc compounded future value
    - FV = PV * (1 + r) ^ n

    Parameters
    ----------
    periods : int
    rate : float
    initial : float, optional
        default 1.0

    Returns
    -------
    float
        final value
    """
    return initial * (1 + rate) ** periods


def ci_daily_monthly(final: float, days: int) -> float:
    """Naive convert daily compound interest rate to monthly

    Parameters
    ----------
    final : float
    days : int

    Returns
    -------
    float
        estimated monthly ci rate
    """
    r_daily = ci_rate(final=final, periods=days)
    return (1 + r_daily) ** (1 * 30)
