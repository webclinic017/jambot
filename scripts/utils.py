import argparse

from jambot import config as cf
from jambot import getlog
from jambot.ml.storage import ModelStorageManager
from jambot.utils.azureblob import BlobStorage
from jambot.utils.secrets import SecretsManager

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
        ModelStorageManager().fit_save_models()

        # also upload current "least_important_cols"
        p = cf.p_data / 'feats'
        BlobStorage(container=p).upload_file(p=p / 'least_imp_cols.pkl')
