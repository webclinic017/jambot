from abc import ABCMeta, abstractmethod

from .__init__ import *


class DictRepr(object):
    """Class to add better string rep with to_dict"""

    def to_dict_str(self):
        """TODO func to convert values of output dicts to string reprs based on dtype"""
        pass

    def __str__(self) -> str:
        if hasattr(self, 'to_dict'):
            data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
            return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))
        else:
            return str(self)

    def __repr__(self) -> str:
        return str(self)


class SignalEvent(object):
    """Class to store functions to perform on signal"""

    def __init__(self, *types):
        """
        types: tuple
            types of arguments signal will emit
        """
        self._funcs = []
        self._types = types

    @property
    def funcs(self):
        return self._funcs

    def emit(self, *args):
        """Emit signal with specified args

        Raises
        ------
        ValueError
            If wrong number of arguments
        ValueError
            If any wrong type of arguments
        """
        types = self._types

        if not len(args) == len(types):
            raise ValueError(f'Wrong number of arguments. Expected: {len(types)}, Received: {len(args)}')

        for i, _type in enumerate(types):
            arg = args[i]
            if not isinstance(arg, _type):
                raise ValueError(f'Incorrect type. Expected: {_type}, Received: {arg}, {type(arg)}')

        for func in self._funcs:
            func(*args)

    def connect(self, func: Callable):
        """Register action to perform on signal emit

        Parameters
        ----------
        func : Callable
            function to call with args on self.emit()
        """
        self._funcs.append(func)


class Observer(DictRepr, metaclass=ABCMeta):
    """Object which will be called every timestep when attached to main stream"""

    def __init__(self, parent_listener=None):
        self._listeners = []
        self._duration = 0
        self.c = None
        self.parent = None

        # NOTE might need to pass c here too?
        if not parent_listener is None:
            parent_listener.attach(self)

    @property
    def timestamp(self):
        return self.c.Index if not self.c is None else None

    @property
    def duration(self):
        """Duration in number of candles/periods"""
        return self._duration

    @property
    def listeners(self):
        return self._listeners

    @property
    def num_listeners(self):
        return len(self.listeners)

    def attach(self, objs: Iterable) -> None:
        """Attach child listener"""

        # enforce iteration
        # NOTE this may be slower for the sake of cleaner code
        if not hasattr(objs, '__iter__'):
            objs = (objs, )

        for obj in objs:
            self.listeners.append(obj)
            obj.parent = self

    def detach(self) -> None:
        """Detach self from parent's listeners"""
        if not self.parent is None:
            self.parent.listeners.remove(self)

    @abstractmethod
    def step(self) -> None:
        """Perform specific actions at each timestep, must be implemmented by each object"""
        raise NotImplementedError('Must implement step in child class!')

    def _step(self, c: tuple) -> None:
        """Increment one timestep, perform actions, call all children's step method"""
        for listener in self._listeners:
            listener._step(c)

        self.c = c
        self._duration += 1
        self.step()

    def to_dict(self):
        return {listener: listener.to_dict() for listener in self.listeners}

    # def __str__(self) -> str:
    #     return str({self: self.to_dict()})

    # def __repr__(self) -> str:
    #     return str(self)


class Clock(Observer):
    """Top level observer to send timesteps using dataframe index"""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self._iterator = df.itertuples()

    def next(self):
        """Move all listeners forward one step"""
        self._step(c=next(self._iterator))

    def step(self):
        """Move clock forward one timestep"""
        pass
