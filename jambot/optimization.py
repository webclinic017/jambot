from pathlib import Path
from datetime import (datetime as dt, timedelta as delta)
from joblib import Parallel, delayed

import pandas as pd

from . import (
    functions as f,
    backtest as bt)

def run_trend(symbol, startdate, df, against, wth, row, titles):
    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])

    # Strat_Trend
    trend = bt.Strat_Trend(speed=(against, wth))
    strats = []
    strats.append(trend)

    sym = bt.Backtest(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    sym.decide_full()

    a = sym.account
    dfTemp.loc[0] = [against, wth, round(a.min,3), round(a.max,3), round(a.balance,3), sym.strats[0].tradecount()]

    return dfTemp

def run_parallel():
    symbol = 'XBTUSD'
    p = f.topfolder / 'data/symbols.csv'
    dfsym = pd.read_csv(p)
    dfsym = dfsym[dfsym.symbol==symbol]
    startdate, daterange = dt(2019, 1, 1), 365 * 3
    dfall = f.read_csv(startdate, daterange, symbol=symbol)

    for row in dfsym.itertuples():
        strattype = 'trendrev'
        norm = (0.004, 0.024)
        syms = Parallel(n_jobs=-1)(delayed(run_single)(strattype, startdate, dfall, speed0, speed1, row, norm) for speed0 in range(6, 27, 1) for speed1 in range(6, 18, 1))

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
        strat.stoppercent = -0.03
        
    elif strattype == 'trend':
        speed = (row.against, row.withspeed)
        strat = bt.Strat_Trend(speed=speed)
        strat.slippage = 0.002

    sym = bt.Backtest(symbol=symbol, startdate=startdate, strats=[strat], df=df, row=row, partial=False)
    sym.decide_full()

    return sym

def run_trendrev(symbol, startdate, df, against, wth, row, titles, norm):   
    # Strat_TrendRev
    strat = bt.Strat_TrendRev(speed=(against, wth), norm=norm)
    sym = bt.Backtest(symbol=symbol, startdate=startdate, strats=[strat], df=df, row=row)
    sym.decide_full()

    a = sym.account

    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])
    dfTemp.loc[0] = [against, wth, round(a.min,3), round(a.max,3), round(a.balance,3), strat.tradecount()]

    return dfTemp

def run_chop(symbol, startdate, df, against, wth, tpagainst, tpwith, lowernorm, uppernorm, row, titles):
    dfTemp = pd.DataFrame(columns=['against', 'wth', 'tpagainst', 'tpwith', 'lowernorm', 'uppernorm', 'min', 'max', 'final', 'numtrades'])

    # lowernorm /= 2
    # uppernorm /= 2
    
    strats = []
    chop = bt.Strat_Chop(speed=(against, wth), speedtp=(tpagainst, tpwith), norm=(lowernorm, uppernorm))
    strats.append(chop)

    sym = bt.Backtest(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    try:
        sym.decide_full()
    except ZeroDivisionError:
        print(symbol, lowernorm, uppernorm)

    a = sym.account
    dfTemp.loc[0] = [against, wth, tpagainst, tpwith, lowernorm, uppernorm, round(a.min,3), round(a.max,3), round(a.balance,3), sym.strats[0].tradecount()]

    return dfTemp