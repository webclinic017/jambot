import argparse

from jambot import config as cf
from jambot import getlog
from jambot.livetrading import ExchangeManager
from jambot.ml.storage import ModelStorageManager

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
