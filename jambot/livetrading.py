from datetime import datetime as dt
from datetime import timedelta as delta
from typing import *

import pandas as pd

from jambot import comm as cm
from jambot import config as cf
from jambot import functions as f
from jambot import getlog
from jambot.exchanges.bitmex import Bitmex
from jambot.exchanges.bybit import Bybit
from jambot.exchanges.exchange import SwaggerExchange
from jambot.ml import models as md
from jambot.ml.storage import ModelStorageManager
from jambot.tradesys import backtest as bt
from jambot.tradesys.strategies import base, ml, sfp
from jambot.utils import google as gg

log = getlog(__name__)


def check_sfp(df):
    """
    NOTE not used

    run every hour
    get last 196 candles, price info only
    create sfp object
    check if current candle returns any sfp objects
    send discord alert with the swing fails, and supporting info
    'Swing High to xxxx', 'swung highs at xxxx', 'tail = xx%'
    if one candle swings highs and lows, go with... direction of candle? bigger tail?
    """

    strat = sfp.Strategy()
    strat.init(df=df)
    sfps = strat.is_swingfail()
    msg = ''
    cdl = strat.cdl
    stypes = dict(high=1, low=-1)

    for k in stypes.keys():
        lst = list(filter(lambda x: k in x['name'], sfps))
        side = stypes[k]

        if lst:
            msg += 'Swing {} to {} | tail = {:.0%}\n'.format(
                k.upper(),
                cdl.getmax(side=side),
                cdl.tailpct(side=side))

            for s in lst:
                msg += '    {} at: {}\n'.format(s['name'], s['price'])

    if msg:
        cm.discord(msg=msg, channel='sfp')


def check_filled_orders(minutes: int = 5, test: bool = True) -> None:
    """Get orders filled in last n minutes and send discord msg

    Parameters
    ----------
    minutes : int, optional
        default 5
    test : bool, optional
        use testnet, default True
    """
    symbol = 'XBTUSD'
    starttime = dt.utcnow() + delta(minutes=minutes * -1)

    # Get discord username for tagging
    df_users = gg.UserSettings().get_df(load_api=True)

    # TODO not done for bybit yet... cant get all orders at once...
    for exch in iter_exchanges(refresh=False, df_users=df_users, exch_name='bitmex'):

        orders = exch.get_filled_orders(starttime=starttime)
        prec = exch.get_instrument(symbol=symbol)['precision']

        # NOTE could filter by symbols eventually
        if orders:
            msg = '\n'.join([o.summary_msg(exch=exch, nearest=prec) for o in orders])

            exch.set_positions()
            current_qty = f.pretty_dict(
                m=dict(current_qty=f'{exch.current_qty(symbol=symbol):+,}'),
                prnt=False,
                bold_keys=True)

            msg = '{}\n{}{}'.format(
                df_users.loc[(exch.exch_name, exch.user)]['discord'],
                cm.py_codeblock(msg),
                current_qty)
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
    gg.OpenOrders(**kw).set_df(exchs=[e for e in f.as_list(exchs) if e.user in ('jayme', 'testnet')])
    gg.UserBalance(**kw).set_df(exchs=exchs)
    gg.TradeHistory(**kw).set_df(strat=strat)

    batcher.run_batch()


def show_current_status(n: int = 30, **kw) -> None:
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
        1: (cf.colors['lightblue'], 'black'),
        -1: (cf.colors['lightred'], 'white')}

    # show last n rows of current active strategy df_pred
    return get_df_pred(**kw)[list(m_fmt.keys())] \
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
    return df.drop(columns=cols).pipe(f.left_merge, df_new[cols])


def get_df_raw(exch_name: str, symbol: str, interval: int = 15) -> pd.DataFrame:
    """get OHLCV df for running strat live

    Parameters
    ----------
    symbol : str, optional
        by default SYMBOL
    interval : int, optional
        by default 15

    Returns
    -------
    pd.DataFrame
    """
    from jambot.database import db

    offset = {1: 16, 15: 4}.get(interval)
    startdate = f.inter_now(interval) + delta(days=-offset)

    return db.get_df(exch_name=exch_name, symbol=symbol, startdate=startdate, interval=interval)


def get_df_pred(
        name: str = 'lgbm',
        test: bool = False,
        **kw) -> pd.DataFrame:
    """Convenicce to get df_pred used by live trading

    Parameters
    ----------
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
    df = get_df_raw(**kw) \
        .pipe(md.add_signals, name=name)

    # load saved/trained models from blob storage and add pred signals
    return ModelStorageManager(test=test) \
        .df_pred_from_models(df=df, name=name)


def iter_exchanges(
        refresh: bool = True,
        df_users: pd.DataFrame = None,
        exch_name: Union[str, List[str]] = None) -> SwaggerExchange:
    """Iterate exchange objs for all users where bot is enabled

    Parameters
    ----------
    refresh : bool
        refresh exch data on init or not
    df_users : pd.DataFrame, optional
        df with user data from google, default None
    exch_name : Union[str, List[str]], optional
        single or multiple exchanges to filter, default None

    Returns
    -------
    Bitmex
        initialized Bitmex exch obj
    """
    if df_users is None:
        df_users = gg.UserSettings().get_df(load_api=True)

    df_users = df_users.query('bot_enabled == True')

    if not exch_name is None:
        df_users = df_users.loc[f.as_list(exch_name)]

    m_exch = dict(bitmex=Bitmex, bybit=Bybit)

    for (exch_name, user), m in df_users.to_dict(orient='index').items():
        test = True if 'test' in user.lower() else False
        Exchange = m_exch[exch_name]

        # NOTE only set up for XBT currently
        yield Exchange(
            user=user,
            test=test,
            refresh=refresh,
            pct_balance=m['xbt'],
            api_key=m['key'],
            api_secret=m['secret'],
            discord=m['discord'])


def run_strat(
        name: str = 'lgbm',
        df_pred: pd.DataFrame = None,
        order_offset: float = -0.0006,
        **kw) -> ml.Strategy:

    # allow passing in to replace OHLC cols and run again
    if df_pred is None:
        df_pred = get_df_pred(name=name, **kw)

    # run strat in "live" mode to get expected state
    strat = ml.make_strat(live=True, order_offset=order_offset, **kw)

    cols = ['open', 'high', 'low', 'close', 'signal']
    bm = bt.BacktestManager(
        startdate=df_pred.index[0],
        strat=strat,
        df=df_pred[cols]).run()

    return strat


def run_strat_live(
        interval: int = 15,
        exch_name: Union[str, List[str]] = None,
        test: bool = False) -> None:
    """Run strategy on given interval and adjust orders
    - run at 15 seconds passed the interval (bitmex OHLC REST delay)

    Parameters
    ----------
    interval : int, default 15
    exch_name: Union[str, List[str]], optional
        limit live exchanges, default None
    """
    name = 'lgbm'

    # run strat for bmex first
    strat_bmex = run_strat(interval=interval, symbol='XBTUSD', name=name, exch_name='bitmex')

    # replace ohlc and run strat for bbit data
    df_bbit = replace_ohlc(
        df=strat_bmex.df,
        df_new=get_df_raw(exch_name='bybit', symbol='BTCUSD', interval=interval))

    strat_bbyt = run_strat(name=name, df_pred=df_bbit, symbol='BTCUSD', exch_name='bybit')

    m_strats = dict(
        bitmex=strat_bmex,
        bybit=strat_bbyt)

    df_users = gg.UserSettings(auth=False).get_df(load_api=True)

    exchs = []
    for exch in iter_exchanges(df_users=df_users, exch_name=exch_name):
        strat = m_strats[exch.exch_name]
        symbol = exch.default_symbol

        # TODO will need to test this with multiple symbols eventually
        exch.reconcile_orders(
            symbol=symbol,
            expected_orders=strat.broker.expected_orders(symbol=symbol, exch=exch),
            test=test)

        exchs.append(exch)

    # write current strat trades/open positions to google
    write_balance_google(strat=strat, exchs=exchs, test=test)
