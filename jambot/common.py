import json
from abc import ABCMeta, abstractmethod

from joblib import Parallel
from tqdm.auto import tqdm


class DictRepr(object, metaclass=ABCMeta):
    """Class to add better string rep with to_dict"""

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


class Serializable(dict, metaclass=ABCMeta):
    """Mixin class to make object json serializeable
    https://stackoverflow.com/questions/18478287/making-object-json-serializable-with-regular-encoder
    """

    def __new__(cls, *args, **kwargs):
        """hack needed so 'items' can be called by json
        - set in __new__ so dont have to explicitely initialize object with __init__ if mixin
        """
        obj = super().__new__(cls)
        obj.__setitem__('dummy', 1)
        return obj

    @abstractmethod
    def __json__(self) -> dict:
        """Return a dict representation of self"""
        raise NotImplementedError('Must specify a "__json__" method.')

    def items(self) -> dict:
        """Called by json.dumps"""
        return self.__json__().items()

    def to_json(self) -> str:
        """Json dump self to string"""
        return json.dumps(self)


class ProgressParallel(Parallel):
    """Modified from https://stackoverflow.com/a/61900501/6278428"""

    def __init__(self, use_tqdm: bool = True, total: int = None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
