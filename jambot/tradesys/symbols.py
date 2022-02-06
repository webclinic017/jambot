import pandas as pd

from jambot import config as cf
from jgutils.logger import getlog  # jambot.__init__ imports from symbols

log = getlog(__name__)


class Symbol(str):
    """Class to represent trading pair eg XBTUSD, with extra params"""
    def __new__(cls, value, *args, **kwargs):
        return super(Symbol, cls).__new__(cls, value)

    def __init__(
            self,
            symbol: str,
            exch_name: str = 'bitmex',
            is_inverse: bool = True,
            lot_size: int = 100,
            tick_size: float = 0.5,
            prec: int = 1,
            **kw):

        self.symbol = symbol
        self.is_inverse = is_inverse
        self.lot_size = lot_size
        self.tick_size = tick_size
        self.prec = prec
        self.exch_name = exch_name

        # only used for display NOTE might need to adjust this
        self.prec_qty = max(0, 4 - self.prec) if not is_inverse else 0

    @property
    def tick_size_str(self) -> str:
        """Convert scientific tick_size eg 1e-5 to decimal repr"""
        return f'{self.tick_size:.16f}'.rstrip('0')

    def __repr__(self) -> str:
        """Create repr with extra params to differentiate from normal string"""
        vals = ['exch_name', 'prec', 'tick_size_str', 'lot_size', 'is_inverse']
        m = {k: getattr(self, k) for k in vals}
        m['tick_size'] = m.pop('tick_size_str')

        strvals = ', '.join(f'{k}={v}' for k, v in m.items())
        return f'<{self.symbol}: {strvals}>'


class Symbols(object):
    """Collection of Symbols, raw data loaded from db"""

    def __init__(self):
        self.p_syms = cf.p_ftr / 'df_symbols.ftr'
        self._df_syms = None  # type: pd.DataFrame
        self._syms = {}  # dype: Dict[str, Symbol]

    def reset(self) -> None:
        """Reset cache"""
        self._syms = {}

    def load_data(self) -> pd.DataFrame:
        """Load df_syms from database

        Returns
        -------
        pd.DataFrame
        """
        log.info('Loading symbol data from database')
        from jambot.tables import Symbols as _Symbols
        return _Symbols().get_df()

    @property
    def df_syms(self) -> pd.DataFrame:
        """df of all symbols per exchange"""
        if self._df_syms is None:
            self._df_syms = self.load_data()

        return self._df_syms

    def _init_symbol(self, symbol: str, exch_name: str) -> Symbol:
        """Create Symbol from self.df_syms data"""
        key = (exch_name, symbol)
        if not key in self.df_syms.index:
            raise IndexError(f'No symbol "{symbol}" found for exchange "{exch_name}"')

        row = self.df_syms.loc[key]  # get row Series from dataframe
        return Symbol(symbol, exch_name=exch_name, **row.to_dict())

    def symbol(self, symbol: str, exch_name: str = 'bitmex', reset: bool = False) -> Symbol:
        """Get Symbol obj, cached or create new

        Parameters
        ----------
        symbol : str
            eg 'XBTUSD'
        exch_name : str, optional
            default 'bitmex'

        Returns
        -------
        Symbol
            instantiated Symbol obj
        """
        key = (exch_name, symbol)

        if key in self._syms.keys() and not reset:
            _symbol = self._syms[key]
        else:
            _symbol = self._init_symbol(symbol, exch_name)
            self._syms[key] = _symbol  # save to symbols cache

        return _symbol
