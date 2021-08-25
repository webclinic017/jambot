import logging
from typing import *

import google.cloud.logging as gcl
import numpy as np
import pandas as pd
import pygsheets
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from pygsheets import Spreadsheet
from pygsheets.client import Client

from jambot import functions as f


class GoogleFormatter(logging.Formatter):
    """Custom formatter for google logging messages
    - Not used currently
    """

    def format(self, record):
        logmsg = super().format(record)
        return dict(
            msg=logmsg,
            args=record.args)


def get_creds(scopes: list = None) -> Credentials:
    from jambot.utils.secrets import SecretsManager
    m = SecretsManager('google_creds.json').load
    return service_account.Credentials.from_service_account_info(m, scopes=scopes)


def get_google_logging_client() -> gcl.Client:
    """Create google logging client

    Returns
    -------
    google.logging.cloud.Client
        logging client to add handler to Python logger
    """
    return gcl.Client(credentials=get_creds())


def add_google_logging_handler(log: logging.Logger) -> None:
    """Add gcl handler to Python logger

    Parameters
    ----------
    log : logging.Logger
    """
    gcl_client = get_google_logging_client()
    handler = gcl_client.get_default_handler()
    handler.setLevel(logging.INFO)
    # handler.setFormatter(GoogleFormatter())
    log.addHandler(handler)


def get_google_client() -> Client:
    """Get google spreadsheets Client

    Returns
    -------
    Client
        pygsheets.client.Client
    """
    scopes = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    return pygsheets.authorize(custom_credentials=get_creds(scopes=scopes))


def get_google_sheet(name: str = 'Jambot Settings', gc: Client = None) -> Spreadsheet:
    """Get google wb object

    Parameters
    ----------
    name : str
        default wb to open, default 'Jambot Settings'
    gc : Client, optional
        google client obj, default None

    Returns
    -------
    pygsheets.Spreadsheet
    """
    gc = gc or get_google_client()
    return gc.open(name)


def ws_df(name: str, wb: Spreadsheet = None) -> pd.DataFrame:
    """Get worksheet as dataframe

    Parameters
    ----------
    name : str
        Google sheets ws name
    wb : pygsheets.Spreadsheet

    Returns
    -------
    pd.DataFrame
    """
    wb = wb or get_google_sheet()

    return wb.worksheet_by_title(name).get_as_df() \
        .pipe(f.lower_cols)


class GoogleSheet():
    index_col = None
    rng_start = (1, 1)

    def __init__(
            self,
            gc: Client = None,
            wb: Spreadsheet = None,
            name: str = None,
            batcher: 'SheetBatcher' = None,
            **kw):
        self.name = name or self.__class__.__name__

        # don't need to init client/wb if using batcher
        if not batcher:
            self.gc = gc or get_google_client()
            self.wb = wb or get_google_sheet(gc=self.gc)

        self.df = None
        self.batcher = batcher

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process df after loading from gs"""
        return df

    def get_df(self, **kw) -> pd.DataFrame:
        df = ws_df(name=self.name, wb=self.wb, **kw)

        if self.index_col:
            df = df.set_index(self.index_col)

        return df \
            .replace({'%': ''}, regex=True) \
            .pipe(self.process_df)

    def set_df(self, df: pd.DataFrame) -> None:
        """Set local df to google sheets

        Parameters
        ----------
        df : pd.DataFrame
        """
        if not self.batcher:
            ws = self.wb.worksheet_by_title(self.name)
            ws.set_dataframe(df=df, start=self.rng_start, nan='')
        else:
            self.batcher.add_df(df=df, rng=self.rng_start)


class SheetBatcher(GoogleSheet):
    def __init__(self, name: str, **kw):
        super().__init__(name=name, **kw)
        self.batches = []

    def add_df(self, df: pd.DataFrame, rng: Tuple[int, int]) -> None:
        self.batches.append([rng, df])

    def run_batch(self) -> None:
        """Combine all dfs in self.batches to larger df, submit to google
        - NOTE dfs must all be in a continuous range
        """
        rows_min, cols_min = float('inf'), float('inf')
        rows_max, cols_max = 0, 0

        # loop to get min/max ranges
        for rng, df in self.batches:
            rows_min = min(rows_min, rng[0] - 1)
            rows_max = max(rows_max, rng[0] + df.shape[0] - 1)
            cols_min = min(cols_min, rng[1] - 1)
            cols_max = max(cols_max, rng[1] + df.shape[1] - 1)

        # create blank df of max shape
        df_out = pd.DataFrame(np.nan, index=range(rows_min, rows_max), columns=range(cols_min, cols_max))

        # loop again to set dfs to df_out
        for rng, df in self.batches:
            # set data in place
            df_out.loc[rng[0] - 1: df.shape[0] - 1, rng[1] - 1: rng[1] + df.shape[1] - 2] = df.values

            # rename columns
            m_rename = {i + rng[1] - 1: df.columns[i] for i in range(df.shape[1])}
            df_out.rename(columns=m_rename, inplace=True)

        # return df_out
        self.rng_start = (rows_min + 1, cols_min + 1)
        self.set_df(df=df_out)
        self.batches = []


class UserSettings(GoogleSheet):
    index_col = 'user'

    def __init__(self, **kw):
        super().__init__(**kw)
        f.set_self(vars())

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df \
            .replace(dict(TRUE=1, FALSE=0)) \
            .astype(float) \
            .assign(bot_enabled=lambda x: x.bot_enabled.astype(bool))

        # NOTE kinda sketch but works for now
        sym_cols = [c for c in df.columns if len(c) == 3]
        df[sym_cols] = df[sym_cols] / 100

        return df


class Bitmex(GoogleSheet):
    def __init__(self, **kw):
        super().__init__(name='Bitmex', **kw)


class TradeHistory(Bitmex):
    rng_start = (1, 15)

    def set_df(self, strat, **kw) -> None:
        """Set df of trade history to gs"""
        cols = ['ts', 'side', 'dur', 'entry', 'exit', 'pnl', 'pnl_acct', 'profitable']
        df = strat.df_trades(last=20)[cols].copy() \
            .assign(**{c: lambda x, c=c: x[c].apply(lambda x: f.percent(x)) for c in ('pnl', 'pnl_acct')}) \
            .pipe(f.remove_underscore)

        super().set_df(df=df, **kw)


class OpenOrders(Bitmex):
    rng_start = (1, 9)

    def set_df(self, exch, **kw) -> None:
        """Set df of trade history to gs"""
        df = exch.df_orders(refresh=True, new_only=True) \
            .rename(columns=dict(order_type='ord_type')) \
            .pipe(f.remove_underscore)

        super().set_df(df=df, **kw)


class OpenPositions(Bitmex):
    rng_start = (1, 1)

    def set_df(self, exch, **kw) -> None:
        m_conv = dict(
            sym='underlying',
            qty='currentQty',
            entry='avgEntryPrice',
            last='lastPrice',
            pnl='unrealisedPnl',
            pnl_pct='unrealisedPnlPcnt',
            roe_pct='unrealisedRoePcnt',
            value='maintMargin')

        data = [{k: pos[v] for k, v in m_conv.items()} for pos in exch._positions.values()]
        df = pd.DataFrame(data=data) \
            .assign(**{c: lambda x, c=c: x[c] / exch.div for c in ('pnl', 'value')}) \
            .pipe(f.remove_underscore)

        super().set_df(df=df, **kw)
