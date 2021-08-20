import logging

import google.cloud.logging as gcl
from google.auth.credentials import Credentials
from google.oauth2 import service_account


class GoogleFormatter(logging.Formatter):
    def format(self, record):
        logmsg = super().format(record)
        return dict(
            msg=logmsg,
            args=record.args)


def get_creds(scopes: list = None) -> Credentials:
    from jambot.utils.secrets import SecretsManager
    m = SecretsManager('google_creds.json').load
    return service_account.Credentials.from_service_account_info(m, scopes=scopes)


def get_google_sheet():
    import pygsheets

    scopes = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    creds = get_creds(scopes=scopes)

    # easiest way loading from json file
    # return pygsheets.authorize(service_account_file=p).open('Jambot Settings')
    return pygsheets.authorize(custom_credentials=creds).open('Jambot Settings')


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
