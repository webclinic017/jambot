from abc import ABCMeta, abstractmethod

from .__init__ import *


class SignalEvent(object):
    """Class to store functions to perform on signal"""

    def __init__(self, *types):
        """
        types: tuple
            types of arguments signal will emit
        """
        self._funcs = []
        self._types = types

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


class Observer(object, metaclass=ABCMeta):
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

    def attach(self, obj: 'Observer', c: tuple = None) -> None:
        """Attach child listener"""
        # NOTE could check to make sure child is instance of listener as well?

        self.listeners.append(obj)
        obj.parent = self

        # call obj.step() here?
        # if not c is None:
        #     obj.step(c)

    def detach(self) -> None:
        """Detach self from parent's listeners"""
        if not self.parent is None:
            self.parent.listeners.remove(self)

    @abstractmethod
    def step(self, c: tuple) -> None:
        """Perform specific actions at each timestep, must be implemmented by each object"""
        raise NotImplementedError('Must implement step in child class!')

    def _step(self, c: tuple) -> None:
        """Increment one timestep, perform actions, call all children's step method"""
        for listener in self._listeners:
            listener._step(c)

        self.c = c
        self._duration += 1
        self.step(c)

    def to_dict(self):
        return {listener: listener.to_dict() for listener in self.listeners}

    def __str__(self) -> str:
        return str({self: self.to_dict()})

    # def __repr__(self) -> str:
    #     return str(self)
