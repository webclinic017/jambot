import numpy as np
import pandas as pd
import pygsheets
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from pygsheets import Spreadsheet
from pygsheets.client import Client

from jambot import functions as f


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
    from jambot.utils.secrets import SecretsManager
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
            .assign(**{c: lambda x, c=c: x[c].apply(lambda x: f.percent(x)) for c in cols}) \



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
                df_out.iloc[row_start - 1: row_end - 1, col_start: col_end] = df.columns.values

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

        m_types = {c: float for c in df.columns} \
            | {
                'bot_enabled': bool,
                'discord': str}

        df = df \
            .replace(dict(TRUE=1, FALSE=0)) \
            .astype(m_types) \
            .assign(discord=lambda x: '<@' + x.discord + '>')

        # NOTE kinda sketch but works for now
        sym_cols = [c for c in df.columns if len(c) == 3]
        df[sym_cols] = df[sym_cols] / 100

        return df


class Bitmex(GoogleSheet):
    def __init__(self, **kw):
        super().__init__(name='Bitmex', **kw)


class TradeHistory(Bitmex):
    rng_start = (1, 15)

    def set_df(self, strat, last: int = 15, **kw) -> None:
        """Set df of trade history to gs"""
        cols = ['ts', 'side', 'dur', 'entry', 'exit', 'pnl', 'pnl_acct', 'profitable', 'status']
        df = strat.df_trades(last=last)[cols].copy() \
            .pipe(self.as_percent, cols=('pnl', 'pnl_acct')) \
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
        df = pd.DataFrame(data=data, columns=list(m_conv.keys())) \
            .assign(**{c: lambda x, c=c: x[c] / exch.div for c in ('pnl', 'value')}) \
            .pipe(self.as_percent, cols=('pnl_pct', 'roe_pct')) \
            .pipe(f.remove_underscore)

        super().set_df(df=df, **kw)


class UserBalance(Bitmex):
    rng_start = (11, 2)

    def set_df(self, exch, **kw) -> None:
        data = dict(upnl=[exch.unrealized_pnl], bal=[exch.total_balance_margin])
        df = pd.DataFrame(data)
        super().set_df(df=df, **kw)
