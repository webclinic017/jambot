from datetime import datetime as dt
from datetime import timedelta as delta
from typing import *

import pandas as pd

from jambot import comm as cm
from jambot import config as cf
from jambot import functions as f
from jambot import getlog
from jambot.common import DictRepr
from jambot.exchanges.bitmex import Bitmex
from jambot.exchanges.bybit import Bybit
from jambot.exchanges.exchange import SwaggerAPIException, SwaggerExchange
from jambot.ml import models as md
from jambot.ml.storage import ModelStorageManager
from jambot.tables import Tickers
from jambot.tradesys import backtest as bt
from jambot.tradesys.strategies import base, ml
from jambot.utils import google as gg
from jgutils import functions as jf
from jgutils import pandas_utils as pu

log = getlog(__name__)


class ExchangeManager(DictRepr):
    m_exch = dict(bitmex=Bitmex, bybit=Bybit)

    def __init__(self, df_users: pd.DataFrame = None):
        self._exchanges = {}

        if df_users is None:
            df_users = gg.UserSettings(auth=False).get_df(load_api=True)

        self.df_users = df_users

    def to_dict(self) -> dict:
        return dict(init_exchanges=len(self.list_exchs))

    @property
    def exchanges(self) -> Dict[Tuple[str, str], SwaggerExchange]:
        """Dict of init exchanges"""
        return self._exchanges

    @property
    def list_exchs(self) -> List[SwaggerExchange]:
        """Get list of all init exchanges

        Returns
        -------
        List[SwaggerExchange]
        """
        return list(self.exchanges.values())

    def default(self, exch_name: str, test: bool = False, **kw) -> SwaggerExchange:
        """Convenience func to get main exch for api data eg funding/tickers

        Parameters
        ----------
        exch_name : str
        test : bool

        Returns
        -------
        SwaggerExchange
        """
        user = 'jayme' if not test else 'testnet'
        return self.get_exch(exch_name, user=user, test=test, **kw)

    def get_exch(self, exch_name: str, user: str, **kw) -> SwaggerExchange:
        """Get single exchange. Will init if not already init.

        Parameters
        ----------
        exch_name : str
        user : str

        Returns
        -------
        SwaggerExchange
        """
        k = (exch_name, user)
        return self.exchanges.get(k, self._init_exch(*k, **kw))

    def _init_exch(self, exch_name: str, user: str, **kw) -> SwaggerExchange:
        """Init single exchange

        Parameters
        ----------
        exch_name : str
        user : str

        Returns
        -------
        SwaggerExchange
        """
        k = (exch_name, user)

        if not k in self.df_users.index:
            raise ValueError(f'User: {k} not in self.df_users')

        m = self.df_users.loc[k].to_dict()

        # NOTE only set up for XBT currently
        Exchange = self.m_exch[exch_name]
        exch = Exchange.from_dict(user=user, m=m, **kw)

        # save to exchange cache
        self.exchanges[k] = exch

        return exch

    def iter_exchanges(
            self,
            refresh: bool = True,
            exch_name: Union[str, List[str]] = None,
            test_exchs: bool = False) -> Iterable[SwaggerExchange]:
        """Iterate exchange objs for all users where bot is enabled

        Parameters
        ----------
        refresh : bool
            refresh exch data on init or not
        exch_name : Union[str, List[str]], optional
            single or multiple exchanges to filter, default None
        test_exchs : bool
            use testnets regardless if enabled or not

        Yields
        ------
        Iterable[SwaggerExchange]
            initialized exch obj
        """
        if not test_exchs:
            df_users = self.df_users.query('bot_enabled == True')
        else:
            df_users = self.df_users.loc[(slice(None), 'testnet'), :]

        # filter to single exchange
        if exch_name:
            df_users = df_users.loc[jf.as_list(exch_name)]

        for (exch_name, user), m in df_users.to_dict(orient='index').items():

            yield self.get_exch(exch_name, user, refresh=refresh)


def check_filled_orders(minutes: int = 5, em: ExchangeManager = None) -> None:
    """Get orders filled in last n minutes and send discord msg

    Parameters
    ----------
    minutes : int, optional
        default 5
    em : ExchangeManager
    """
    symbol = 'XBTUSD'
    starttime = dt.utcnow() + delta(minutes=minutes * -1)

    # Get discord username for tagging
    em = em or ExchangeManager()

    # TODO not done for bybit yet... cant get all orders at once...
    for exch in em.iter_exchanges(refresh=False, exch_name='bitmex'):

        orders = exch.get_filled_orders(starttime=starttime)
        prec = exch.get_instrument(symbol=symbol)['precision']

        # NOTE could filter by symbols eventually
        if orders:
            msg = '\n'.join([o.summary_msg(exch=exch, nearest=prec) for o in orders])

            exch.set_positions()
            current_qty = jf.pretty_dict(
                m=dict(current_qty=exch.current_pos_msg(symbol)),
                prnt=False,
                bold_keys=True)

            msg = '{}\n{}\n{}'.format(
                em.df_users.loc[(exch.exch_name, exch.user)]['discord'],
                current_qty,
                cm.py_codeblock(msg))

            cm.discord(msg=msg, channel='orders')


def write_balance_google(
        strat: base.StrategyBase,
        exchs: Union[SwaggerExchange, List[SwaggerExchange]],
        test: bool = False,
        gc: gg.Client = None) -> None:
    """Update google sheet "Bitmex" with current strat performance


    Parameters
    ----------
    strat : base.StrategyBase
    exchs : Union[SwaggerExchange, List[SwaggerExchange]]
    test : bool, optional
        only display dfs, don't write to sheet, default False
    """
    gc = gc or gg.get_google_client()
    batcher = gg.SheetBatcher(gc=gc, name='Bitmex', test=test)
    kw = dict(batcher=batcher)

    gg.OpenPositions(**kw).set_df(exchs=exchs)
    gg.OpenOrders(**kw).set_df(exchs=[e for e in jf.as_list(exchs) if e.user in ('jayme', 'testnet')])
    gg.UserBalance(**kw).set_df(exchs=exchs)
    gg.TradeHistory(**kw).set_df(strat=strat)

    batcher.run_batch()


def show_current_status(exch_name: str, symbol: str, n: int = 30, **kw) -> None:
    """Show current live strategy signals

    # NOTE this is kinda extra, should probs just show strat's recent trades
    """
    from jambot.utils.styles import highlight_val

    # set columns and num format
    cols_ohlc = ['open', 'high', 'low', 'close']
    m_fmt = {c: '{:,.0f}' for c in cols_ohlc}
    m_fmt |= dict(
        rolling_proba='{:.3f}',
        signal='{:.0f}')

    # set colors to highlight signal column
    m_color = {
        1: (cf.COLORS['lightblue'], 'black'),
        -1: (cf.COLORS['lightred'], 'white')}

    # show last n rows of current active strategy df_pred
    return get_df_pred(exch_name=exch_name, symbol=symbol, **kw)[list(m_fmt.keys())] \
        .tail(n) \
        .style.format(m_fmt) \
        .apply(highlight_val, subset=['signal'], m=m_color)


def replace_ohlc(df: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Replace existing ohlc candles with new

    Parameters
    ----------
    df : pd.DataFrame
        df to replace
    df_new : pd.DataFrame
        df to get new ohlc from

    Returns
    -------
    pd.DataFrame
    """
    cols = ['open', 'high', 'low', 'close']
    return df.drop(columns=cols).pipe(pu.left_merge, df_new[cols])


def get_df_raw(
        exch_name: str,
        symbol: str,
        interval: int = 15,
        funding_exch: SwaggerExchange = None) -> pd.DataFrame:
    """get OHLCV df for running strat live

    Parameters
    ----------
    symbol : str, optional
        by default SYMBOL
    interval : int, optional
        by default 15
    funding_exch : SwaggerExchange, optional
        pass in exch to use to get latest funding data

    Returns
    -------
    pd.DataFrame
    """

    offset = {1: 16, 15: 4}[interval]
    startdate = f.inter_now(interval) + delta(days=-offset)

    return Tickers().get_df(
        exch_name=exch_name,
        symbol=symbol,
        startdate=startdate,
        interval=interval,
        funding=True if exch_name == 'bitmex' else False,
        funding_exch=funding_exch)


def get_df_pred(
        exch_name: str,
        symbol: str,
        name: str = 'lgbm',
        test: bool = False,
        **kw) -> pd.DataFrame:
    """Convenicce to get df_pred used by live trading

    Parameters
    ----------
    exch_name: str
    symbol: str
    name : str, optional
        model name, by default 'lgbm'
    test : bool
        use test or live models on azure blob storage

    Returns
    -------
    pd.DataFrame
        df with preds added
    """

    # add signals
    df = get_df_raw(exch_name=exch_name, symbol=symbol, **kw) \
        .pipe(md.add_signals, name=name, symbol=symbol, use_important_dynamic=True)

    # load saved/trained models from blob storage and add pred signals
    return ModelStorageManager(test=test) \
        .df_pred_from_models(df=df, name=name)


def run_strat(
        name: str = 'lgbm',
        df_pred: pd.DataFrame = None,
        order_offset: float = -0.0006,
        exch_name: str = 'bitmex',
        symbol: str = cf.SYMBOL,
        **kw) -> ml.Strategy:

    # allow passing in to replace OHLC cols and run again
    if df_pred is None:
        df_pred = get_df_pred(name=name, exch_name=exch_name, symbol=symbol, **kw)

    # run strat in "live" mode to get expected state
    strat = ml.make_strat(live=True, order_offset=order_offset, exch_name=exch_name, symbol=symbol, **kw)

    cols = ['open', 'high', 'low', 'close', 'signal']
    bt.BacktestManager(
        startdate=df_pred.index[0],
        strat=strat,
        df=df_pred[cols]).run()

    return strat


def run_strat_live(
        interval: int = 15,
        exch_name: Union[str, List[str]] = None,
        test: bool = False,
        em: ExchangeManager = None,
        test_models: bool = False,
        test_exchs: bool = False) -> None:
    """Run strategy on given interval and adjust orders
    - run at 15 seconds passed the interval (bitmex OHLC REST delay)

    Parameters
    ----------
    interval : int, default 15
    exch_name: Union[str, List[str]], optional
        limit live exchanges, default None
    em : ExchangeManager, optional
    test : bool
        don't submit orders, just print
    test_models : bool, optional
        use models from 'jambot-app-test' not 'jambot-app', default False
        - Must have recently fit/saved models with ModelStorageManager
    test_exchs : bool, optional
        only use test exchanges (eg testnets)
    """
    name = 'lgbm'
    em = em or ExchangeManager()

    # run strat for bmex first
    strat_bmex = run_strat(
        interval=interval,
        symbol='XBTUSD',
        name=name,
        exch_name='bitmex',
        funding_exch=em.get_exch('bitmex', 'jayme'),
        test=test_models)

    # replace ohlc and run strat for bbit data
    df_bbit = replace_ohlc(
        df=strat_bmex.df,
        df_new=get_df_raw(exch_name='bybit', symbol='BTCUSD', interval=interval))

    strat_bbit = run_strat(name=name, df_pred=df_bbit, symbol='BTCUSD', exch_name='bybit')

    m_strats = dict(
        bitmex=strat_bmex,
        bybit=strat_bbit)

    for exch in em.iter_exchanges(exch_name=exch_name, test_exchs=test_exchs):
        strat = m_strats[exch.exch_name]
        symbol = exch.default_symbol

        # TODO will need to test this with multiple symbols eventually
        try:
            exch.reconcile_orders(
                symbol=symbol,
                expected_orders=strat.broker.expected_orders(symbol=symbol, exch=exch),
                test=test)
        except SwaggerAPIException as e:
            e.send_error_discord()

    # write current strat trades/open positions to google
    write_balance_google(strat=strat, exchs=em.list_exchs, test=test)
