"""Simple file encryption management to avoid storing plaintext passwords"""
import json
from io import StringIO
from pathlib import Path

import pandas as pd
import yaml
from cryptography.fernet import Fernet

from jambot import config as cf

# log = getlog(__name__)  # SecretsManager cant use logger, its used to init logger


class SecretsManager(object):
    """Context manager to handle loading encrypted files from secrets folder

    Parameters
    ----------
    load_filename : str
        Init with filename to load, call with .load property

    Examples
    ---
    >>> from jambot.utils.secrets import SecretsManager

    - load yaml dict automatically
    >>> m = SecretsManager('db.yaml').load

    - as context manager:
    >>> with SecretsManager('db.yaml') as file:
        m = yaml.full_load(file)

    - re encrypt all files in /unencrypted after pws change
    >>> SecretsManager().encrypt_all_secrets()
    """

    def __init__(self, load_filename: Path = None):
        self.p_secret = cf.p_sec
        self.p_key = self.p_secret / 'jambot.key'
        self.p_unencrypt = cf.p_proj / '_unencrypted'
        self.load_filename = load_filename

    def __enter__(self):
        return self.get_secret_file(name=self.check_file)

    def __exit__(self, *args):
        return

    @property
    def key(self):
        """Load key from secrets dir"""
        return open(self.p_key, 'rb').read()

    @property
    def check_file(self):
        name = self.load_filename
        if name is None:
            raise AttributeError('Must init SecretsManager with file to load!')

        return name

    @property
    def load(self):
        """Load file, auto parse if known extension"""
        name = self.check_file
        ext = name.split('.')[-1]  # get file extension eg '.yaml'

        file = self.get_secret_file(name=name)

        if ext == 'yaml':
            return yaml.load(file, Loader=yaml.Loader)
        elif ext == 'csv':
            return pd.read_csv(self.from_bytes(file))
        elif ext == 'json':
            return json.loads(file)

        return file

    def get_secret_file(self, name):
        """Get file from secrets folder by name and decrypt"""
        p = self.p_secret / name
        if not p.exists():
            raise FileNotFoundError(f'Couldn\'t find secret file: {name}')

        return self.decrypt_file(p=p, key=self.key)

    def encrypt_all_secrets(self):
        """Convenience func to auto encrypt everything in _unencrypted with jambot.key
        - Use to re-encrypt after pw changes (every three months for email/sap)"""
        i = 0
        for p in self.p_unencrypt.glob('*'):
            if self.encrypt_file(p):
                i += 1

        print(f'Successfully encrypted [{i}] file(s).')

    def encrypt_file(self, p: Path, **kw):
        """Open file, encrypt, and save it to secrets folder"""
        if not p.exists():
            raise FileNotFoundError(f'Couldn\'t find secret file: {p}')

        with open(p, 'rb') as file:
            file_data = file.read()

        self.write(file_data, name=p.name, **kw)
        return True

    def write(self, file_data: bytes, name: str, p_save: Path = None, **kw):
        """Write unencrypted file back as encrypted

        Parameters
        ----------
        file_data : bytes
            File to encrypt and write back to secrets folder
        name : str
            filename including extension, eg credentials.yaml
        p_save : Path
            optional path to save encrypted file
        - NOTE only handles writing yaml files so far, need to add csv, txt ect
        """
        if p_save is None:
            p_save = self.p_secret

        p_save = p_save / name
        ext = name.split('.')[-1]

        # non-bytes dict passed back, encode as bytes here
        if ext in ('yaml', 'yml') and isinstance(file_data, dict):
            file_data = yaml.dump(file_data).encode()  # encode str as bytes

        fn = Fernet(self.key)
        encrypted_data = fn.encrypt(file_data)

        with open(p_save, 'wb') as file:
            file.write(encrypted_data)

    def decrypt_file(self, p, key):
        """Decrypt file and return, DO NOT save back to disk"""
        fn = Fernet(key)
        with open(p, 'rb') as file:
            encrypted_data = file.read()

        decrypted_data = fn.decrypt(encrypted_data)
        return decrypted_data

    def write_key(self):
        """Generates a key and save it into a file"""
        key = Fernet.generate_key()

        with open(self.p_key, 'wb') as key_file:
            key_file.write(key)

    def from_bytes(self, bytes: bytes) -> str:
        """Return string from bytes object
        - Useful for reading csv/excel data from bytes so far"""
        result = str(bytes, 'UTF-8')
        return StringIO(result)
