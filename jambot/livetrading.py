from datetime import datetime as dt
from datetime import timedelta as delta

import pandas as pd

from jambot import SYMBOL
from jambot import config as cf
from jambot import functions as f
from jambot.exchanges.bitmex import Bitmex
from jambot.ml import models as md
from jambot.ml.storage import ModelStorageManager
from jambot.tradesys import backtest as bt
from jambot.tradesys.strategies import base, ml, sfp, trendrev
from jambot.utils import google as gg


def compare_state(strat, pos):
    # return TRUE if side is GOOD
    # Could also check current qty?
    # only works for trend, don't use for now
    qty = pos['currentQty']
    side = f.side(qty)

    ans = True if side == 0 or strat.get_side() == side else False

    if not ans:
        err = '{}: {}, expected: {}'.format(strat.bm.symbolshort, side, strat.get_side())
        f.discord(err)

    return ans


def refresh_gsheet_balance(u=None):
    sht = f.get_google_sheet()
    ws = sht.worksheet_by_title('Bitmex')
    df = ws.get_as_df(start='A1', end='J15')
    lst = list(df['Sym'].dropna())
    syms = []

    p = cf.p_res / 'symbols.csv'
    df2 = pd.read_csv(p)
    for row in df2.itertuples():
        if row.symbolshort in lst:
            syms.append(bt.BacktestManager(symbol=row.symbol, row=row))

    # if u is None:
    #     u = User()
    # write_balance_google(syms, u, sht=sht, ws=ws, df=df)


def check_sfp(df):
    # run every hour
    # get last 196 candles, price info only
    # create sfp object
    # check if current candle returns any sfp objects
    # send discord alert with the swing fails, and supporting info
    # 'Swing High to xxxx', 'swung highs at xxxx', 'tail = xx%'
    # if one candle swings highs and lows, go with... direction of candle? bigger tail?

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
        f.discord(msg=msg, channel='sfp')


def check_filled_orders(minutes: int = 5, exch: Bitmex = None, test: bool = True) -> None:
    """Get orders filled in last n minutes and send discord msg

    Parameters
    ----------
    minutes : int, optional
        default 5
    exch : Bitmex, optional
    test : bool, optional
        use testnet, default True
    """
    starttime = dt.utcnow() + delta(minutes=minutes * -1)

    for exch in iter_exchanges(refresh=False):

        orders = exch.get_filled_orders(starttime=starttime)
        prec = exch.get_instrument(symbol=SYMBOL)['tickSize']

        # NOTE could filter by symbols eventually

        if orders:
            msg = '\n'.join([o.summary_msg(exch=exch, nearest=prec) for o in orders])

            exch.set_positions()
            current_qty = f.pretty_dict(
                m=dict(current_qty=f'{exch.current_qty(symbol=SYMBOL):+,}'),
                prnt=False,
                bold_keys=True)

            msg = f'{f.py_codeblock(msg)}{current_qty}\n@here'
            f.discord(msg=msg, channel='orders')


def old_write_balance_google(syms, u, sht=None, ws=None, preservedf=False, df=None):

    if sht is None:
        sht = f.get_google_sheet()
    if ws is None:
        ws = sht.worksheet_by_title('Bitmex')

    if df is None:
        if not preservedf:
            df = pd.DataFrame(columns=['Sym', 'Size', 'Entry', 'Last', 'Pnl', '%',
                              'ROE', 'Value', 'Dur', 'Conf'], index=range(14))
        else:
            df = ws.get_as_df(start='A1', end='J15')

    u.set_positions()

    for i, bm in enumerate(syms):
        symbol = bm.symbolbitmex
        figs = bm.decimal_figs
        pos = u.get_position(symbol)

        df.at[i, 'Sym'] = bm.symbolshort
        if bm.tradingenabled:
            df.at[i, 'Size'] = pos['currentQty']
            df.at[i, 'Entry'] = round(pos['avgEntryPrice'], figs)
            df.at[i, 'Last'] = round(pos['lastPrice'], figs)
            df.at[i, 'Pnl'] = round(pos['unrealisedPnl'] / u.div, 3)
            df.at[i, '%'] = f.percent(pos['unrealisedPnlPcnt'])
            df.at[i, 'ROE'] = f.percent(pos['unrealisedRoePcnt'])
            df.at[i, 'Value'] = pos['maintMargin'] / u.div

        if bm.strats:
            strat = bm.strats[0]
            t = strat.trades[-1]
            df.at[i, 'Dur'] = t.duration()
            df.Conf[i] = t.conf

    # set profit/balance
    u.set_total_balance()
    df.at[9, 'Size'] = u.unrealized_pnl
    df.at[9, 'Entry'] = u.total_balance_margin

    # set funding rate
    rate, hrs = u.funding_rate()
    df.at[12, 'Sym'] = 'Funding:'
    df.at[12, 'Size'] = f.percent(rate)
    df.at[12, 'Entry'] = hrs

    # set current time
    df.at[13, 'Sym'] = 'Last:'
    df.at[13, 'Size'] = dt.strftime(dt.utcnow(), f.time_format(mins=True))

    # concat last 10 trades for google sheet
    bm = list(filter(lambda x: x.symbol == 'XBTUSD', syms))[0]
    if bm.strats:
        dfTrades = bm.strat.df_trades(last=10).drop(columns=['N', 'Contracts', 'Bal'])
        dfTrades.timestamp = dfTrades.timestamp.dt.strftime('%Y-%m-%d %H')
        dfTrades.Pnl = dfTrades.Pnl.apply(lambda x: f.percent(x))
        dfTrades.PnlAcct = dfTrades.PnlAcct.apply(lambda x: f.percent(x))
    else:
        dfTrades = ws.get_as_df(start='Q1', end='Y14')  # df.loc[:, 'timestamp':'PnlAcct']

    df = pd.concat([df, u.df_orders(refresh=True), dfTrades], axis=1)
    ws.set_dataframe(df, (1, 1), nan='')
    # return df


def old_run_toploop(u=None, partial=False, dfall=None):
    # run every 1 hour, or when called by check_filled_orders()
    from jambot.database import db

    # Google - get user/position info
    sht = f.get_google_sheet()
    g_usersettings = sht.worksheet_by_title('UserSettings').get_all_records()
    p = cf.p_res / 'symbols.csv'
    dfsym = pd.read_csv(p)
    g_user = g_usersettings[0]
    syms = []

    # Bitmex - get user/position info
    if u is None:
        # u = User()
        u = None

    u.set_positions()
    u.set_orders()
    u.reserved_balance = g_user['Reserved Balance']  # could just pass g_user to User()

    # TODO: filter dfall to only symbols needed, don't pull everything from db
    # use 'WHERE symbol in []', try pypika
    # Only using XBTUSD currently
    startdate = f.timenow() + delta(days=-15)
    if dfall is None:
        dfall = db.get_df(symbol=SYMBOL, startdate=startdate, interval=1)

    for row in dfsym.itertuples():
        if not row.symbol == 'XBTUSD':
            continue
        try:
            # match google user with bitmex position, add %balance
            weight = float(g_user[row.symbolshort].strip('%')) / 100
            pos = u.get_position(row.symbolbitmex)
            pos['percent_balance'] = weight

            symbol = row.symbol
            # NOTE may need to set timestamp/symbol index when using more than just XBTUSD
            df = dfall[dfall.symbol == symbol]  # .reset_index(drop=True)

            # TREND_REV
            speed = (16, 6)
            norm = (0.004, 0.024)
            strat = trendrev.Strategy(speed=speed, norm=norm)
            strat.stop_pct = -0.03
            strats = [strat]

            bm = bt.BacktestManager(symbol=symbol, startdate=startdate, strats=strats,
                                    row=row, df=df, partial=partial, u=u)
            if weight <= 0:
                bm.tradingenabled = False  # this should come from strat somehow
            bm.decide_full()
            syms.append(bm)

            if bm.tradingenabled:
                actual = u.get_orders(bm.symbolbitmex, bot_only=True)
                theo = strat.final_orders(u, weight)

                # matched, missing, not_matched = compare_orders(theo, actual, show=True)

                # u.cancel_bulk(not_matched)
                # u.amend_bulk(validate_matched(matched, show=True))
                # u.place_bulk(missing)

        except:
            f.send_error(symbol)

    write_balance_google(syms, u, sht)


def write_balance_google(strat: base.StrategyBase, exch: Bitmex) -> None:
    """Update google sheet "Bitmex" with current strat performance

    Parameters
    ----------
    strat : base.StrategyBase
    exch : Bitmex
    """
    gc = gg.get_google_client()
    batcher = gg.SheetBatcher(gc=gc, name='Bitmex')
    kw = dict(batcher=batcher)

    gg.OpenPositions(**kw).set_df(exch=exch)
    gg.OpenOrders(**kw).set_df(exch=exch)
    gg.UserBalance(**kw).set_df(exch=exch)
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


def get_df_pred(
        symbol: str = SYMBOL,
        interval: int = 15,
        name: str = 'lgbm',
        test: bool = False) -> pd.DataFrame:
    """Convenicce to get df_pred used by live trading

    Parameters
    ----------
    symbol : str, optional
        by default SYMBOL
    interval : int, optional
        by default 15
    name : str, optional
        model name, by default 'lgbm'
    test : bool
        use test or live models on azure blob storage

    Returns
    -------
    pd.DataFrame
        df with preds added
    """
    from jambot.database import db

    # load raw data from db and add signals
    offset = {1: 16, 15: 4}.get(interval)
    startdate = f.timenow(interval) + delta(days=-offset)
    df = db.get_df(symbol=symbol, startdate=startdate, interval=interval) \
        .pipe(md.add_signals, name=name)

    # load saved/trained models from blob storage and add pred signals
    return ModelStorageManager(test=test) \
        .df_pred_from_models(df=df, name=name)


def iter_exchanges(refresh: bool = True) -> Bitmex:
    """Iterate exchange objs for all users

    Parameters
    ----------
    refresh : bool
        refresh exch data on init or not

    Returns
    -------
    Bitmex
        initialized Bitmex exch obj
    """
    df_users = gg.UserSettings().get_df() \
        .query('bot_enabled == True')

    for user, m in df_users.to_dict(orient='index').items():
        test = True if 'test' in user.lower() else False

        # NOTE only set up for XBT currently
        yield Bitmex(user=user, test=test, refresh=refresh, pct_balance=m['xbt'])


def run_strat(interval: int = 15, symbol: str = SYMBOL, name: str = 'lgbm', **kw) -> ml.Strategy:
    df_pred = get_df_pred(symbol, interval, name, **kw)

    # run strat in "live" mode to get expected state
    strat = ml.make_strat(live=True)

    cols = ['open', 'high', 'low', 'close', 'signal']
    bm = bt.BacktestManager(
        symbol=symbol,
        startdate=df_pred.index[0],
        strat=strat,
        df=df_pred[cols]).run()

    return strat


def run_strat_live(interval: int = 15) -> None:
    """Run strategy on given interval and adjust orders
    - run at 15 seconds passed the interval (bitmex OHLC REST delay)

    Parameters
    ----------
    interval : int, default 15
    """
    symbol = SYMBOL
    name = 'lgbm'
    strat = run_strat(interval=interval, symbol=symbol, name=name)

    m_exch = {}
    for exch in iter_exchanges():
        m_exch[exch.user] = exch

        exch.reconcile_orders(
            symbol=symbol,
            expected_orders=strat.broker.expected_orders(symbol=symbol, exch=exch))

    # write current strat trades/open positions to google
    write_balance_google(strat=strat, exch=m_exch['jayme'])
