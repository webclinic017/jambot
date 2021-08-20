import argparse

from jambot import getlog
from jambot.utils.secrets import SecretsManager

log = getlog(__name__)
cli = argparse.ArgumentParser()

cli.add_argument(
    '--encrypt_creds',
    default=False,
    action='store_true',
    help='Re-encrypt credentials')


if __name__ == '__main__':
    a = cli.parse_args()

    if a.encrypt_creds:
        SecretsManager().encrypt_all_secrets()
