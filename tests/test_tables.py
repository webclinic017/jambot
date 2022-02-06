from datetime import datetime as dt
from datetime import timedelta as delta

from pytest import fixture

from jambot.livetrading import ExchangeManager
from jambot.tables import Funding, Tickers


@fixture(scope='session')
def ticker() -> Tickers:
    """Test init Tickers tbl"""
    return Tickers()


@fixture(scope='session')
def funding() -> Funding:
    """Test init Funding tbl"""
    return Funding()


def test_tickers(ticker: Tickers, em: ExchangeManager):
    """Test can load table with funding"""
    df = ticker.get_df(
        symbol='XBTUSD',
        interval=15,
        startdate=dt.now() + delta(days=-90),
        # funding=True,
        funding_exch=em.default('bitmex'))

    assert df.shape[1] == 6, 'Incorrect number of columns loaded'
    assert df.shape[0] >= 1, 'No OHLCV data loaded from database'


def test_update_exchange(ticker: Tickers, funding: Funding, em: ExchangeManager):
    """Test tables can call update and get data from exchange
    - NOTE could probably test this more detailed
    - Only updating default data
    """
    exchs = [em.default(k) for k in ('bitmex', 'bybit')]

    ticker.update_from_exch(exchs=exchs, interval=15, test=True)
    funding.update_from_exch(exchs=exchs[0], symbols='XBTUSD', test=True)
