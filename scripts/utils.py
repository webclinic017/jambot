import argparse

from jambot import config as cf
from jambot import data, getlog
from jambot.livetrading import ExchangeManager
from jambot.ml.models import DEFAULT_SIGNALS
from jambot.ml.storage import ModelStorageManager
from jambot.signals import SignalManager
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
    # action='store_true',
    help='Run skopt optimization')

cli.add_argument(
    '--n_jobs',
    type=int,
    default=-1,
    # action='store_true',
    help='n_jobs for skopt')


if __name__ == '__main__':
    a = cli.parse_args()

    if a.encrypt_creds:
        SecretsManager().encrypt_all_secrets()
    elif a.fit_models:

        # fit models and upload
        em = ExchangeManager()
        ModelStorageManager().fit_save_models(em=em)

        # also upload current "least_important_cols"
        p = cf.p_data / 'feats'
        for prefix in ('most', 'least'):
            BlobStorage(container=p).upload_file(p=p / f'{prefix}_imp_cols.pkl')

    elif a.skopt:
        df_all = SignalManager.default() \
            .add_signals(df=data.default_df(), signals=DEFAULT_SIGNALS)

        log.info(f'df_all: {df_all.shape}')

        mfm = MlflowManager()
        res = sko.run_opt(n_calls=a.skopt, df=df_all, mfm=mfm, n_jobs=a.n_jobs)
