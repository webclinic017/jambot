import json
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import *

from joblib import Parallel

from jambot import config as cf

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    pass


class DictRepr(object, metaclass=ABCMeta):
    """Class to add better string rep with to_dict"""

    def to_dict(self) -> dict:
        pass

    def to_dict_str(self):
        """TODO func to convert values of output dicts to string reprs based on dtype"""
        pass

    def __str__(self) -> str:
        """Create string representation of self from dict or list of strings"""
        data = []

        if hasattr(self, 'to_dict'):
            m = self.to_dict()

            # convert list to dict of self items
            if isinstance(m, (list, tuple)):
                m = {k: getattr(self, k) for k in m}

            data = ['{}={}'.format(k, v) for k, v in m.items()]

        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self) -> str:
        return str(self)

    def keys(self) -> list:
        return self.to_dict().keys()

    def __getitem__(self, key: str) -> Any:
        """Used to call dict() on object
        - NOTE this calls self.to_dict() for every key requested
        - NOTE not actually used yet
        """
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


class Serializable(metaclass=ABCMeta):
    """Mixin class to make object json serializeable
    https://stackoverflow.com/questions/18478287/making-object-json-serializable-with-regular-encoder
    """

    @abstractmethod
    def __json__(self) -> dict:
        """Return a dict representation of self"""
        raise NotImplementedError('Must specify a "__json__" method.')

    def items(self) -> dict:
        """Called by json.dumps"""
        return self.__json__().items()

    def to_json(self) -> str:
        """Json dump self to string
        - NOTE this only works with dicts now, not sure if other objs ever needed
        """
        return json.dumps({k: v for k, v in self.__json__().items()})


class ProgressParallel(Parallel):
    """Modified from https://stackoverflow.com/a/61900501/6278428"""

    def __init__(self, use_tqdm: bool = True, total: int = None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self.bar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}'  # limit bar width in terminal
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
                disable=not self._use_tqdm,
                total=self._total,
                bar_format=self.bar_format) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class DynConfig(metaclass=ABCMeta):
    """Class to allow quick instantiation from dynamically written config values"""
    log_keys = abstractproperty()

    @classmethod
    def from_config(cls, symbol: str = cf.SYMBOL, **kw):
        """Instantiate from dynamic config file"""
        kw = cf.dynamic_cfg(symbol, keys=cls.log_keys) | kw
        return cls(**kw)
