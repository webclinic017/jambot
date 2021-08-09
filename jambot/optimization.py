from datetime import datetime as dt
from datetime import timedelta as delta
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from . import backtest as bt
from . import functions as f


def run_trend(symbol, startdate, df, against, wth, row, titles):
    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])

    # Strat_Trend
    trend = bt.Strat_Trend(speed=(against, wth))
    strats = []
    strats.append(trend)

    bm = bt.BacktestManager(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    bm.decide_full()

    a = bm.account
    dfTemp.loc[0] = [against, wth, round(a.min, 3), round(a.max, 3), round(a.balance, 3), bm.strats[0].tradecount()]

    return dfTemp


def run_parallel():
    symbol = 'XBTUSD'
    p = f.p_res / 'symbols.csv'
    dfsym = pd.read_csv(p)
    dfsym = dfsym[dfsym.symbol == symbol]
    startdate, daterange = dt(2019, 1, 1), 365 * 3
    dfall = f.read_csv(startdate, daterange, symbol=symbol)

    for row in dfsym.itertuples():
        strattype = 'trendrev'
        norm = (0.004, 0.024)
        syms = Parallel(n_jobs=-1)(delayed(run_single)(strattype, startdate, dfall, speed0, speed1, row, norm)
                                   for speed0 in range(6, 27, 1) for speed1 in range(6, 18, 1))

    return syms


def run_single(strattype, startdate, dfall, speed0, speed1, row=None, norm=None, symbol=None):
    import backtest as bt

    if not row is None:
        symbol = row.symbol
    df = f.filter_df(dfall, symbol)

    speed = (speed0, speed1)

    if strattype == 'trendrev':
        strat = bt.Strat_TrendRev(speed=speed, norm=norm)
        strat.slippage = 0
        strat.stop_pct = -0.03

    elif strattype == 'trend':
        speed = (row.against, row.withspeed)
        strat = bt.Strat_Trend(speed=speed)
        strat.slippage = 0.002

    bm = bt.BacktestManager(symbol=symbol, startdate=startdate, strats=[strat], df=df, row=row, partial=False)
    bm.decide_full()

    return bm


def run_trendrev(symbol, startdate, df, against, wth, row, titles, norm):
    # Strat_TrendRev
    strat = bt.Strat_TrendRev(speed=(against, wth), norm=norm)
    bm = bt.BacktestManager(symbol=symbol, startdate=startdate, strats=[strat], df=df, row=row)
    bm.decide_full()

    a = bm.account

    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])
    dfTemp.loc[0] = [against, wth, round(a.min, 3), round(a.max, 3), round(a.balance, 3), strat.tradecount()]

    return dfTemp


def run_chop(symbol, startdate, df, against, wth, tpagainst, tpwith, lowernorm, uppernorm, row, titles):
    dfTemp = pd.DataFrame(columns=['against', 'wth', 'tpagainst', 'tpwith',
                          'lowernorm', 'uppernorm', 'min', 'max', 'final', 'numtrades'])

    # lowernorm /= 2
    # uppernorm /= 2

    strats = []
    chop = bt.Strat_Chop(speed=(against, wth), speedtp=(tpagainst, tpwith), norm=(lowernorm, uppernorm))
    strats.append(chop)

    bm = bt.BacktestManager(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    try:
        bm.decide_full()
    except ZeroDivisionError:
        print(symbol, lowernorm, uppernorm)

    a = bm.account
    dfTemp.loc[0] = [against, wth, tpagainst, tpwith, lowernorm, uppernorm, round(
        a.min, 3), round(a.max, 3), round(a.balance, 3), bm.strats[0].tradecount()]

    return dfTemp
