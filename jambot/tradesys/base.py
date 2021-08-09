from abc import ABCMeta, abstractmethod

from ..common import DictRepr
from .__init__ import *

log = getlog(__name__)


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

    def __init__(
            self,
            parent_listener: 'Observer' = None,
            insert_before: bool = False):

        self._listeners = []
        self._duration = 0
        self.c = None
        self.parent = None
        self.timestamp_start = None

        # add as a listener, or swap parent/child order
        if not parent_listener is None:
            if not insert_before:
                parent_listener.attach_listener(self)
            else:
                # NOTE this isn't used yet
                parent_listener.detach_from_parent()

                if parent_listener.has_parent:
                    parent_listener.parent.attach_listener(self)

                self.attach_listener(parent_listener)

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

    @property
    def has_parent(self):
        return not self.parent is None

    def get_parent(self, name: str) -> 'Observer':
        """Get parent in heirarchy by class name

        Parameters
        ----------
        name : str
            class name
        """
        if self.parent is None:
            raise AttributeError(f'Couldn\'t find parent "{name}" in obj tree.')

        cls_name = self.parent.__class__.__name__
        if cls_name == name:
            return self.parent
        else:
            return self.parent.get_parent(name)

    def on_attach(self):
        """Called by parent when child is attached
        - Perform any necessary actions on attach"""
        pass

    def attach_listeners(self, objs: Iterable) -> None:
        """Attach multiple listeners"""
        for obj in objs:
            self.attach_listener(obj)

    def attach_listener(self, obj: object) -> None:
        """Attach child listener"""

        # enforce iteration
        # NOTE this may be slower for the sake of cleaner code
        # if not hasattr(objs, '__iter__'):
        #     objs = (objs, )

        # NOTE check, slower, not necessarily needed
        if obj in self.listeners:
            raise RuntimeError(f'Listener "{obj}" already attached!')

        # for obj in objs:
        self.listeners.append(obj)
        obj.parent = self
        obj.c = self.c
        obj.timestamp_start = self.timestamp
        obj.on_attach()

    def detach_from_parent(self) -> None:
        """Detach self from parent's listeners"""
        if not self.parent is None:
            self.parent.listeners.remove(self)

    @abstractmethod
    def step(self) -> None:
        """Perform specific actions at each timestep, must be implemmented by each object"""
        raise NotImplementedError('Must implement step in child class!')

    def step_clock(self, c: tuple) -> None:
        """Update self and all listener clocks

        Parameters
        ----------
        c : tuple
            named tuple from dataframe.itertuples()
        """
        self.c = c
        self._duration += 1

        for listener in self._listeners:
            listener.step_clock(c)

    def _step(self) -> None:
        """perform actions, call all children's step method in bottom -> top order"""

        for listener in self._listeners:
            listener._step()

        self.step()

    def listener_tree(self, max_depth: int, depth: int = 0) -> dict:
        if depth >= max_depth:
            return

        return {str(self): [child.listener_tree(
            max_depth=max_depth,
            depth=depth + 1) for child in self.listeners]}

    def show_tree(self, max_depth: int = 10) -> None:
        """Show listener Tree

        Parameters
        ----------
        max_depth : int, optional
            max depth of listeners to print, by default 10
        """
        from jambot.sklearn_utils import pretty_dict
        pretty_dict(self.listener_tree(max_depth=max_depth))


class Clock(Observer):
    """Top level observer to send timesteps using dataframe index"""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self._iterator = df.itertuples()

    def run(self):
        """Loop through entire dataframe"""
        for c in self.df.itertuples():
            self.step_clock(c)
            self._step()

    def next(self):
        """Move all listeners forward one step (for testing)"""
        self.step_clock(c=next(self._iterator))
        self._step()

    def step(self):
        """Move clock forward one timestep"""
        pass
