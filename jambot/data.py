"""
Loading dataframe
"""

from collections import defaultdict as dd
from datetime import datetime as dt
from datetime import timedelta as delta
from pathlib import Path
from typing import *

import pandas as pd

from jambot import SYMBOL, ExchSymbols
from jambot import config as cf
from jambot import getlog
from jambot.tables import Tickers
from jgutils import functions as jf
from jgutils import pandas_utils as pu

log = getlog(__name__)


class DataManager(object):
    """Manage loading single/multi symbol dfs from db and feather"""

    def __init__(self):
        self.p_ftr = cf.p_data / 'feather'
        self.dfs_raw = {}  # OHLCVF data
        self.dfs_signals = {}  # data with signals added

    def _get_path(self, symbol: str, exch_name: str) -> Path:
        """Get local ftr path for symbol/exchange

        Parameters
        ----------
        symbol : str
        exch_name : str

        Returns
        -------
        Path
            path to .ftr
        """
        return self.p_ftr / f'{exch_name.lower()}_{symbol.lower()}.ftr'

    def _check_refresh_ftr(self, symbol: str, exch_name: str) -> bool:
        """Check if local ftr needs to be refreshed

        Parameters
        ----------
        symbol : str
        exch_name : str

        Returns
        -------
        bool
            if ftr needs refresh
        """
        p = self._get_path(symbol, exch_name)
        return not p.exists() or dt.fromtimestamp(p.stat().st_mtime) < dt.now() + delta(days=-1)

    def save_df(self, df: pd.DataFrame, symbol: str, exch_name: str) -> Path:
        """Save df to feather file with dropped symbol index

        Parameters
        ----------
        df : pd.DataFrame
        symbol : str
        exch_name : str

        Returns
        -------
        Path
            path of saved file
        """
        p = self._get_path(symbol, exch_name)

        # remove symbol from index for saving to file
        if 'symbol' in df.index.names:
            df = df.droplevel('symbol')  # type: pd.DataFrame

        df.reset_index(drop=False).to_feather(p)
        return p

    @staticmethod
    def load_misc_alts() -> List[str]:
        """Get list of active misc alts from config
        - NOTE might need to make this a bit more dynamic
            - eg option to load from google?
        """
        return cf.MULTI_ALTS_LIST

    def load_from_db(
            self,
            symbols: ExchSymbols,
            interval: int = 15,
            startdate: dt = cf.D_LOWER,
            **kw) -> pd.DataFrame:

        log.info(f'Loading {dict(symbols)} from db')
        dfs = []  # type: List[pd.DataFrame]

        for exch_name, _symbols in symbols.items():
            df = Tickers().get_df(
                symbols={exch_name: _symbols},
                startdate=startdate,
                interval=interval,
                funding=True,
                # funding_exch=em.default('bitmex', refresh=False)
                **kw
            ) \
                .pipe(pu.append_list, dfs)

            # save individual dfs to ftr
            for symbol, _df in df.groupby('symbol'):
                # bit arbitrary, but just avoid saving small dfs, or df with close col only
                if _df.shape[1] > 4:
                    if _df.shape[0] > 1000:
                        self.save_df(_df, symbol, exch_name)
                    else:
                        log.warning(f'No rows to save for {symbol}, {exch_name}')

        df_out = pd.concat(dfs)

        return df_out

    def load_from_local(self, symbols: ExchSymbols) -> pd.DataFrame:
        """Load df from local ftr file

        Parameters
        ----------
        symbols : ExchSymbols

        Returns
        -------
        pd.DataFrame
        """
        log.info(f'Loading {dict(symbols)} from .ftr')
        dfs = []  # type: List[pd.DataFrame]

        for exch_name, _symbols in symbols.items():
            for symbol in jf.as_list(_symbols):
                p = self._get_path(symbol, exch_name)

                df = pd.read_feather(p) \
                    .assign(symbol=symbol) \
                    .assign(symbol=lambda x: x.symbol.astype('category')) \
                    .set_index(['symbol', 'timestamp']) \
                    .pipe(pu.append_list, dfs)

        return pd.concat(dfs)

    def get_df(
            self,
            symbols: ExchSymbols,
            local_only: bool = False,
            db_only: bool = False,
            **kw) -> pd.DataFrame:

        m_reload = dd(list)  # load from db
        m_local = dd(list)  # load local ftr
        dfs = []  # type: List[pd.DataFrame]

        # pass symbols or symbol/exchane??
        for exch_name, _symbols in symbols.items():
            if _symbols == cf.MULTI_ALTS:
                _symbols = self.load_misc_alts()

            for symbol in jf.as_list(_symbols):

                # check if local data needs refresh
                if db_only or (self._check_refresh_ftr(symbol, exch_name) and not local_only):
                    m_reload[exch_name].append(str(symbol))
                else:
                    # load from ftr
                    m_local[exch_name].append(str(symbol))

        if m_reload:
            df_db = self.load_from_db(symbols=m_reload, **kw) \
                .pipe(pu.append_list, dfs)

        if m_local:
            df_local = self.load_from_local(symbols=m_local) \
                .pipe(pu.append_list, dfs)

        dfs_out = pd.concat(dfs)

        # log msg for symbols + min/max timestamps loaded
        m = dfs_out \
            .reset_index(drop=False) \
            .groupby('symbol', as_index=False) \
            .agg(dict(timestamp=['min', 'max'])) \
            .astype(str) \
            .to_records()

        s = '\n\t' + '\n\t'.join([f'{r[1] + ":":<12}{r[2]} - {r[3]}' for r in m])
        log.info(f'Loaded df: [{dfs_out.shape[0]:,.0f}] {s}')

        return dfs_out


def default_df(symbol: str = SYMBOL, exch_name: str = 'bitmex') -> pd.DataFrame:
    """Get simple default df for testing"""
    log.info(f'Loading df {symbol}, {exch_name} from file')
    p = cf.p_data / f'feather/df_{exch_name.lower()}_{symbol.lower()}.ftr'

    # TODO set index symbol/timestamp
    return pd.read_feather(p) \
        .set_index('timestamp')
