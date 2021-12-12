from typing import *

import numpy as np
import pandas as pd
import pygsheets
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from jgutils import functions as jf
from jgutils import pandas_utils as pu
from pygsheets import Spreadsheet
from pygsheets.client import Client

from jambot import functions as f
from jambot.exchanges.exchange import SwaggerExchange


def get_creds(scopes: list = None) -> Credentials:
    """Create google oauth2 credentials

    Parameters
    ----------
    scopes : list, optional
        default None

    Returns
    -------
    Credentials
    """
    from jgutils.secrets import SecretsManager
    m = SecretsManager('google_creds.json').load
    return service_account.Credentials.from_service_account_info(m, scopes=scopes)


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


def ws_df(name: str, wb: Spreadsheet = None, **kw) -> pd.DataFrame:
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
    sheet_id = '1_dAMoMLNYPB0Z6YsMedg4RReOrJsYAzwzrAJgNlS0v0'

    def __init__(
            self,
            gc: Client = None,
            wb: Spreadsheet = None,
            name: str = None,
            batcher: 'SheetBatcher' = None,
            test: bool = False,
            auth: bool = True,
            **kw):
        self.name = name or self.__class__.__name__

        # don't need to init client/wb if using batcher
        if not batcher and auth:
            self.gc = gc or get_google_client()
            self.wb = wb or get_google_sheet(gc=self.gc)

        self.df = None
        self.batcher = batcher
        self.test = test
        self.auth = auth

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process df after loading from gs"""
        return df

    def get_df(self, **kw) -> pd.DataFrame:
        if self.auth:
            df = ws_df(name=self.name, wb=self.wb, **kw)
        else:
            # url for faster reading of sheet without authentication (~0.2s instead of 1.5)
            url = 'https://docs.google.com/spreadsheets/d/' \
                + f'{self.sheet_id}/gviz/tq?tqx=out:csv&sheet={self.name}'
            df = pd.read_csv(url).pipe(f.lower_cols)

        if self.index_col:
            df = df.set_index(self.index_col)

        return df \
            .replace({'%': ''}, regex=True) \
            .pipe(self.process_df, **kw)

    def set_df(self, df: pd.DataFrame) -> None:
        """Set local df to google sheets

        Parameters
        ----------
        df : pd.DataFrame
        """
        if not self.batcher:
            if not self.test:
                ws = self.wb.worksheet_by_title(self.name)
                ws.set_dataframe(df=df, start=self.rng_start, nan='')
            else:
                from IPython.display import display
                display(df)
        else:
            self.batcher.add_df(df=df, gs=self)

    def as_percent(self, df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
        """Convert df columns to % strings for google sheets

        Parameters
        ----------
        df : pd.DataFrame
        cols : tuple

        Returns
        -------
        pd.DataFrame
        """
        return df \
            .assign(**{c: lambda x, c=c: x[c].apply(lambda x: f.percent(x)) for c in cols})


class SheetBatcher(GoogleSheet):
    def __init__(self, name: str, **kw):
        super().__init__(name=name, **kw)
        self.batches = []

    def add_df(self, df: pd.DataFrame, gs: GoogleSheet) -> None:
        self.batches.append([gs, df])

    def run_batch(self) -> None:
        """Combine all dfs in self.batches to larger df, submit to google
        - NOTE dfs must all be in a continuous range
        """
        rows_min, cols_min = float('inf'), float('inf')
        rows_max, cols_max = 0, 0

        # loop to get min/max ranges
        for gs, df in self.batches:
            rng = gs.rng_start
            rows_min = min(rows_min, rng[0] - 1)
            rows_max = max(rows_max, rng[0] + df.shape[0] - 1)
            cols_min = min(cols_min, rng[1] - 1)
            cols_max = max(cols_max, rng[1] + df.shape[1] - 1)

        # create blank df of max shape
        df_out = pd.DataFrame(np.nan, index=range(rows_min, rows_max), columns=range(cols_min, cols_max))

        # loop again to set dfs to df_out
        for gs, df in self.batches:
            # set data in place
            rng = gs.rng_start
            row_start = rng[0] - 1
            row_end = rng[0] + df.shape[0] - 1
            col_start = rng[1] - 1
            col_end = rng[1] + df.shape[1] - 1

            df_out.iloc[row_start: row_end, col_start: col_end] = df.values

            # rename columns if first row dfs
            if row_start == 0:
                m_rename = {i + rng[1] - 1: df.columns[i] for i in range(df.shape[1])}
                df_out.rename(columns=m_rename, inplace=True)
            else:
                # add col header in df as cells
                df_out.iloc[row_start - 1: row_start, col_start: col_end] = df.columns.values

        # return df_out
        self.rng_start = (rows_min + 1, cols_min + 1)
        self.set_df(df=df_out)
        self.batches = []


class UserSettings(GoogleSheet):
    index_col = ['exchange', 'user']

    def __init__(self, auth: bool = False, **kw):
        super().__init__(auth=auth, **kw)
        jf.set_self()

    def process_df(self, df: pd.DataFrame, load_api: bool = False) -> pd.DataFrame:

        m_types = {c: float for c in df.columns} | \
            {
                'bot_enabled': bool,
                'discord': str}

        df = df \
            .replace(dict(TRUE=1, FALSE=0)) \
            .astype(m_types) \
            .assign(discord=lambda x: np.where(x.discord.apply(lambda y: len(y) == 18), '<@' + x.discord + '>', None))

        # NOTE kinda sketch but works for now
        sym_cols = [c for c in df.columns if len(c) == 3]
        df[sym_cols] = df[sym_cols] / 100

        if load_api:
            from jambot.tables import ApiKeys
            df = df.pipe(pu.left_merge, ApiKeys().get_df())

        return df


class Bitmex(GoogleSheet):
    def __init__(self, **kw):
        super().__init__(name='Bitmex', **kw)

    def add_blank_rows(self, df: pd.DataFrame, last: int) -> pd.DataFrame:
        """Add blank rows to end of df to overwrite on google sheet"""
        return df.append(pd.DataFrame(index=range(last - len(df))), ignore_index=True)


class TradeHistory(Bitmex):
    rng_start = (1, 17)

    def set_df(self, strat, last: int = 20, **kw) -> None:
        """Set df of trade history to gs"""
        cols = ['side', 'dur', 'entry', 'exit', 'pnl', 'pnl_acct', 'profitable', 'status']
        df = strat.df_trades(last=last)[cols].copy() \
            .pipe(self.as_percent, cols=('pnl', 'pnl_acct')) \
            .pipe(f.remove_underscore) \
            .reset_index(drop=False) \
            .pipe(self.add_blank_rows, last=last) \
            .assign(timestamp=lambda x: x.timestamp.dt.strftime('%Y-%m-%d %H:%M'))

        super().set_df(df=df, **kw)


class OpenOrders(Bitmex):
    rng_start = (1, 11)

    def set_df(self, exchs: Union[SwaggerExchange, List[SwaggerExchange]], **kw) -> None:
        """Set df of trade history to gs"""
        dfs = []

        for exch in jf.as_list(exchs):
            df = exch.df_orders(refresh=True, new_only=True) \
                .rename(columns=dict(order_type='ord_type')) \
                .pipe(f.remove_underscore) \
                .pipe(pu.append_list, dfs)

        df = pd.concat(dfs) \
            .pipe(self.add_blank_rows, last=10)

        super().set_df(df=df, **kw)


class OpenPositions(Bitmex):
    rng_start = (1, 1)

    def set_df(self, exchs: Union[SwaggerExchange, List[SwaggerExchange]], **kw) -> None:

        dfs = []
        items = ['sym_short', 'qty', 'entry_price', 'last_price', 'u_pnl', 'pnl_pct', 'roe_pct', 'value']
        m_rename = dict(
            sym_short='sym',
            u_pnl='pnl',
            entry_price='entry')

        for exch in jf.as_list(exchs):
            data = [dict(user=exch.user, exch=exch.exch_name)
                    | {k: p.get(k, None) for k in items} for p in exch.positions.values() if p['qty']]

            df = pd.DataFrame(data=data, columns=['user', 'exch'] + items) \
                .pipe(self.as_percent, cols=('pnl_pct', 'roe_pct')) \
                .rename(columns=m_rename) \
                .pipe(f.remove_underscore) \
                .pipe(pu.append_list, dfs)

        df = pd.concat(dfs) \
            .pipe(self.add_blank_rows, last=10)

        super().set_df(df=df, **kw)


class UserBalance(Bitmex):
    rng_start = (11, 1)

    def set_df(self, exchs: Union[SwaggerExchange, List[SwaggerExchange]], **kw) -> None:
        dfs = []

        for exch in jf.as_list(exchs):
            data = dict(
                user=exch.user,
                exch=exch.exch_name,
                upnl=exch.unrealized_pnl,
                balance=exch.total_balance_margin)

            df = pd.DataFrame(data, index=[0]) \
                .pipe(pu.append_list, dfs)

        df = pd.concat(dfs) \
            .pipe(self.add_blank_rows, last=10)

        super().set_df(df=df, **kw)
