from abc import ABCMeta, abstractmethod


class Exchange(object, metaclass=ABCMeta):
    """Base object to represent an exchange connection"""

    def __init__(self, user: str, test: bool = False, **kw):

        self._creds = self.load_creds(user=user)
        self._client = self.init_client(test=test)

    @property
    def client(self):
        """Client connection to exchange"""
        return self._client

    @property
    def key(self):
        return self._creds['key']

    @property
    def secret(self):
        return self._creds['secret']

    @abstractmethod
    def load_creds(self):
        """Load api key and secret from file"""
        pass

    @abstractmethod
    def init_client(self):
        """Initialize client connection"""
        pass

    @abstractmethod
    def refresh(self):
        """Refresh position/orders/etc from exchange"""
        pass

    @abstractmethod
    def set_orders(self):
        """Load/save recent orders from exchange"""
        pass
