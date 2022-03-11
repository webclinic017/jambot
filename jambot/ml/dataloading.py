
import gc
from datetime import datetime as dt
from typing import *

import numpy as np
import pandas as pd

from jambot import config as cf
from jambot import getlog
from jambot.common import DictRepr
from jgutils import fileops as jfl

if TYPE_CHECKING:
    from pathlib import Path

log = getlog(__name__)


class DiskSequence(DictRepr):
    """Class to represent single block of data saved to disk"""

    def __init__(
            self,
            block_index: int,
            index: Union[pd.Index, pd.MultiIndex],
            parent: 'DiskDataFrame',
            p: 'Path'):

        self._len = len(index)
        self.block_index = block_index
        self.parent = parent
        self.p = p
        self.index = index
        self.d_upper_limit = None
        self.batch_size = self.parent.batch_size

    def to_dict(self) -> dict:
        return dict(idx=self.block_index, len=len(self))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: Union[int, slice, List[int]]) -> np.ndarray:
        """Get single or multiple rows of data with [] syntax, same as numpy array
        - ensure parent manages active block, only one block ever in memory

        Parameters
        ----------
        idx : Union[int, slice, List[int]]

        Returns
        -------
        np.ndarray
        """
        parent = self.parent
        if not parent.active_block_idx == self.block_index:
            parent.active_block = self.load_block()
            parent.active_block_idx = self.block_index

        return parent.active_block[idx]

    def set_upper_limit(self, d: dt) -> None:
        """Set max date of values per block to load
        - NOTE this includes d in the values returned

        Parameters
        ----------
        d : dt
        """

        self.d_upper_limit = d
        s = self.index.get_level_values('timestamp')

        # set new len based on count of records not excluded by upper date limit
        self._len = len(s[s <= d])

    def load_block(self) -> np.ndarray:
        """Load single chunk and set int idx

        Parameters
        ----------
        i : int
            block index

        Returns
        -------
        np.ndarray
            array filtered to self max date limit
        """
        # lgbm dataset from samples needs values to be np.float64
        return pd.read_feather(self.p, use_threads=False) \
            .set_index(self.parent.idx_cols) \
            .loc[pd.IndexSlice[:, :self.d_upper_limit], :] \
            .values.astype(np.float64)


class DiskDataFrame(DictRepr):
    """Save df to disk in chunks for reduced memory access"""

    def __init__(self, batch_size: int = 50_000, clear: bool = True):

        self.p_data = cf.p_data / 'diskdf'
        self.batch_size = batch_size
        jfl.check_path(p=self.p_data)

        self.df_queue = pd.DataFrame()
        self.df_index = pd.DataFrame()

        self.idx_cols = []  # type: List[str]
        self.max_idx = 0
        self.active_block = None
        self.active_block_idx = -1

        self.seqs = []  # type: List[DiskSequence]

        if clear:
            self.clear()

    def to_dict(self) -> List[str]:
        return ['batch_size', 'max_idx']

    def clear(self) -> None:
        """Delete all .ftr files
        """
        jfl.clean_dir(p=self.p_data)

    def __len__(self):
        # NOTE not used
        return sum(len(seq) for seq in self.seqs)

    def init_lgbm_seqs(self) -> None:
        """Convert Seqs to lightgbm Seqs
        - this triggers scipy import
        """
        from jambot.ml.sequence import SequenceWrapper

        _seqs = []
        for s in self.seqs:
            seq = SequenceWrapper(s.block_index, s.index, s.parent, s.p)
            seq.d_upper_limit = s.d_upper_limit
            seq._len = s._len
            _seqs.append(seq)

        self.seqs = _seqs

    @property
    def p_chunks(self) -> List['Path']:
        # NOTE not used, chunks managed in seqs now
        return sorted(self.p_data.glob('*'), key=lambda x: int(x.stem))

    def set_upper_limit(self, d: dt) -> None:
        """Set upper limit for all children seqs

        Parameters
        ----------
        d : dt

        Raises
        ------
        TypeError
        """
        if not isinstance(d, dt):
            raise TypeError(f'date must be {dt} not {type(d)}.')

        for seq in self.seqs:
            seq.set_upper_limit(d)

        log.info(f'set upper_limit: {d}')

    def _write_chunk(self, df: pd.DataFrame) -> None:
        """Write single df with shape self.batch_size to .ftr

        Parameters
        ----------
        df : pd.DataFrame
        """

        # set index cols on first write
        if not self.idx_cols:
            self.idx_cols = list(df.index.names)

        # save original index df mapped to integer locs
        rng_idx = range(self.max_idx, len(df) + self.max_idx)
        _df_index = pd.DataFrame(
            data=dict(idx=rng_idx),
            index=df.index)

        self.df_index = pd.concat([self.df_index, _df_index])
        self.max_idx += len(df)
        self._len = self.max_idx
        # log.info(f'write_chunk: max_idx={self.max_idx}, {df.shape}')

        p = self.p_data / f'{self.max_idx - 1}.ftr'
        seq = DiskSequence(block_index=len(self.seqs), index=df.index, p=p, parent=self)
        self.seqs.append(seq)

        df.reset_index(drop=False).to_feather(p)

    def dump_final(self) -> None:
        self._write_chunk(df=self.df_queue)

        self.df_queue = None
        del self.df_queue
        gc.collect()

    def add_chunk(self, df: pd.DataFrame, dump: bool = False) -> None:
        """Add df chunk to .ftr files on disk

        Parameters
        ----------
        df : pd.DataFrame
        dump : bool, optional
            dump remainder chunk immediately after all others, default False
        """
        # add new data to queue
        self.df_queue = pd.concat([self.df_queue, df])

        while len(self.df_queue) >= self.batch_size:
            self._write_chunk(df=self.df_queue.iloc[:self.batch_size])

            # remove chunk just written
            self.df_queue = self.df_queue.iloc[self.batch_size:]

        # dump last rows < batch_size
        if dump:
            self.dump_final()
