import logging

import google.cloud.logging as gcl
import pandas as pd
import pygsheets
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from pygsheets import Spreadsheet

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


def get_google_sheet(name: str = 'Jambot Settings') -> Spreadsheet:
    """Get google Spreadsheet obj

    Parameters
    ----------
    name : str
        default wb to open, default 'Jambot Settings'

    Returns
    -------
    pygsheets.Spreadsheet
    """
    scopes = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    creds = get_creds(scopes=scopes)

    # easiest way loading from json file
    # return pygsheets.authorize(service_account_file=p).open('Jambot Settings')
    return pygsheets.authorize(custom_credentials=creds).open(name)


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

    def __init__(self, wb: Spreadsheet = None, **kw):
        name = self.__class__.__name__
        wb = wb or get_google_sheet()

        f.set_self(vars())

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_df(self, **kw) -> pd.DataFrame:
        df = ws_df(name=self.name, **kw)

        if self.index_col:
            df = df.set_index(self.index_col)

        return df \
            .replace({'%': ''}, regex=True) \
            .pipe(self.process_df)


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
