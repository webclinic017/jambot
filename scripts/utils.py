import argparse

from jambot import config as cf
from jambot import data, getlog
from jambot.livetrading import ExchangeManager
from jambot.ml.models import DEFAULT_SIGNALS
from jambot.ml.storage import ModelStorageManager
from jambot.signals import SignalManager
from jambot.tradesys.symbols import Symbols
from jambot.utils import skopt as sko
from jambot.utils.mlflow import MlflowManager

if True:
    from jgutils.azureblob import BlobStorage
    from jgutils.secrets import SecretsManager

log = getlog(__name__)
cli = argparse.ArgumentParser()

cli.add_argument(
    '--encrypt_creds',
    default=False,
    action='store_true',
    help='Re-encrypt credentials')

cli.add_argument(
    '--fit_models',
    default=False,
    action='store_true',
    help='Fit/save models for last 3 days, upload to azure')

cli.add_argument(
    '--skopt',
    type=int,
    default=200,
    help='Run skopt optimization')

cli.add_argument(
    '--n_jobs',
    type=int,
    default=-1,
    help='n_jobs for skopt')

cli.add_argument(
    '--symbol',
    type=str,
    default='XBTUSD',
    help='symbol to optimize')

cli.add_argument(
    '--exch_name',
    type=str,
    default='bitmex',
    help='exchange to use for symbol data')


if __name__ == '__main__':
    a = cli.parse_args()

    if a.encrypt_creds:
        SecretsManager().encrypt_all_secrets()
    elif a.fit_models:

        # fit models and upload
        em = ExchangeManager()
        # TODO run for multiple symbols
        ModelStorageManager().fit_save_models(em=em, overwrite_all=True, symbol=a.symbol)

        # also upload current "least_important_cols"
        p = cf.p_data / 'feats'
        for prefix in ('most', 'least'):
            BlobStorage(container=p).upload_file(p=p / f'{prefix}_imp_cols.pkl')

    elif a.skopt:
        syms = Symbols()
        symbol = syms.symbol(a.symbol, exch_name=a.exch_name)
        df_all = SignalManager.default() \
            .add_signals(df=data.default_df(symbol=symbol), signals=DEFAULT_SIGNALS)

        log.info(f'df_all: {df_all.shape}')

        mfm = MlflowManager()
        res = sko.run_opt(
            n_calls=a.skopt, df=df_all, mfm=mfm, n_jobs=a.n_jobs, symbol=symbol)
