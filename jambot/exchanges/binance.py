from datetime import datetime as dt
from datetime import timezone as tz
from typing import *

import pandas as pd
from binance.spot import Spot as Client

from jambot import functions as f
from jambot import getlog
from jambot.exchanges.exchange import Exchange
from jambot.tradesys.symbols import Symbol
from jgutils import functions as jf

log = getlog(__name__)

SYMBOL = Symbol('BTCUSD')


class Binance(Exchange):
    base_url = 'https://api3.binance.com'

    def __init__(self, **kw):
        super().__init__(**kw)

    def init_client(self, test: bool = False, **kw):
        return Client(key=self.key, secret=self.secret, base_url=self.base_url)

    def refresh(self):
        pass

    def set_orders(self):
        """Load/save recent orders from exchange"""
        pass

    def get_candles(
            self,
            starttime: dt,
            symbol: str = SYMBOL,
            interval: int = 15,
            limit: int = 1000,
            endtime: dt = None,
            max_pages: float = float('inf'),
            **kw) -> pd.DataFrame:

        _interval = {1: '1h', 15: '15m'}[interval]

        dfs = []  # type: List[pd.DataFrame]
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        num_cols = len(cols)

        for symbol in jf.as_list(symbol):
            pages = 0
            data = []
            _starttime = int(starttime.replace(tzinfo=tz.utc).timestamp())  # convet dt to timestamp int
            _endtime = int((
                min(endtime or dt.utcnow(), f.inter_now(interval))
                .replace(tzinfo=tz.utc)).timestamp())

            while _starttime < _endtime and pages < max_pages:
                pages += 1
                try:
                    _data = self._client.klines(
                        symbol=symbol,
                        interval=_interval,
                        startTime=int(str(_starttime) + '000'),
                        limit=limit)

                    if not _data:
                        log.warning('no data')
                        break

                    data.extend(_data)
                    last_starttime = int(str(_data[-1][0])[:-3])  # last timestamp, strip ms '000'
                    temp_starttime = dt.fromtimestamp(last_starttime, tz.utc) + f.inter_offset(interval)  # add offset
                    _starttime = int(temp_starttime.timestamp())  # back to int
                    log.info(f'{symbol}: starttime={temp_starttime}, len_data={len(_data)}')
                except Exception as e:
                    log.warning(e)
                    log.warning(
                        'Failed to get candle data. '
                        + f'starttime: {_starttime}, endtime: {endtime},len_data: {len(data)}')
                    break

            data = [d[:num_cols] for d in data][:-1]  # only keep needed cols, drop last partial candle
            dtypes = {c: float for c in cols[1:]} \
                | dict(interval=int, symbol=str, volume=int)

            df = pd.DataFrame(data=data, columns=cols) \
                .assign(
                    timestamp=lambda x: pd.to_datetime(x.timestamp, unit='ms'),
                    interval=interval,
                    symbol=symbol,
                    volume=lambda x: x.volume.astype(float))[['interval', 'symbol'] + cols] \
                .astype(dtypes)

            dfs.append(df)

        return pd.concat(dfs)
