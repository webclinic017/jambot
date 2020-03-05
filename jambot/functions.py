# General Functions module
import json
import os
from collections import defaultdict
from datetime import datetime as date
from datetime import timedelta as delta
from pathlib import Path
from sys import platform
from time import time
from urllib import parse as prse

import pandas as pd
import pyodbc
import pypika as pk
import sqlalchemy as sa
import yaml
from dateutil.parser import parse
from pypika import functions as fn

import jambotclasses as c
import livetrading as live

try:
    from IPython.display import display
except ModuleNotFoundError:
    pass

global topfolder
topfolder = Path(__file__).parent

# PARALLEL
def filterdf(dfall, symbol):
    return dfall[dfall.Symbol==symbol].reset_index(drop=True)

def runtrend(symbol, startdate, df, against, wth, row, titles):
    import jambotclasses as c
    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])

    # Strat_Trend
    trend = c.Strat_Trend(speed=(against, wth))
    strats = []
    strats.append(trend)

    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    sym.decidefull()

    a = sym.account
    dfTemp.loc[0] = [against, wth, round(a.min,3), round(a.max,3), round(a.balance,3), sym.strats[0].tradecount()]

    return dfTemp

def run_parallel():
    from joblib import Parallel, delayed
    symbol = 'XBTUSD'
    p = Path(topfolder) / 'data/symbols.csv'
    dfsym = pd.read_csv(p)
    dfsym = dfsym[dfsym.symbol==symbol]
    startdate, daterange = date(2019, 1, 1), 365 * 3
    dfall = readcsv(startdate, daterange, symbol=symbol)

    for row in dfsym.itertuples():
        strattype = 'trendrev'
        norm = (0.004, 0.024)
        syms = Parallel(n_jobs=-1)(delayed(run_single)(strattype, startdate, dfall, speed0, speed1, row, norm) for speed0 in range(6, 27, 1) for speed1 in range(6, 18, 1))

    return syms

def run_single(strattype, startdate, dfall, speed0, speed1, row=None, norm=None, symbol=None):
    import jambotclasses as c

    if not row is None:
        symbol = row.symbol
    df = filterdf(dfall, symbol)

    speed = (speed0, speed1)

    if strattype == 'trendrev':
        strat = c.Strat_TrendRev(speed=speed, norm=norm)
        strat.slippage = 0
        strat.stoppercent = -0.03
        
    elif strattype == 'trend':
        speed = (row.against, row.withspeed)
        strat = c.Strat_Trend(speed=speed)
        strat.slippage = 0.002

    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=[strat], df=df, row=row, partial=False)
    sym.decidefull()

    return sym

def runtrendrev(symbol, startdate, df, against, wth, row, titles, norm):
    import jambotclasses as c
    
    # Strat_TrendRev
    strat = c.Strat_TrendRev(speed=(against, wth), norm=norm)
    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=[strat], df=df, row=row)
    sym.decidefull()

    a = sym.account

    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])
    dfTemp.loc[0] = [against, wth, round(a.min,3), round(a.max,3), round(a.balance,3), strat.tradecount()]

    return dfTemp

def runchop(symbol, startdate, df, against, wth, tpagainst, tpwith, lowernorm, uppernorm, row, titles):
    import jambotclasses as c
    dfTemp = pd.DataFrame(columns=['against', 'wth', 'tpagainst', 'tpwith', 'lowernorm', 'uppernorm', 'min', 'max', 'final', 'numtrades'])

    # lowernorm /= 2
    # uppernorm /= 2
    
    strats = []
    chop = c.Strat_Chop(speed=(against, wth), speedtp=(tpagainst, tpwith), norm=(lowernorm, uppernorm))
    strats.append(chop)

    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    try:
        sym.decidefull()
    except ZeroDivisionError:
        print(symbol, lowernorm, uppernorm)

    a = sym.account
    dfTemp.loc[0] = [against, wth, tpagainst, tpwith, lowernorm, uppernorm, round(a.min,3), round(a.max,3), round(a.balance,3), sym.strats[0].tradecount()]

    return dfTemp


# DATETIME
def checkDate(d):
    if type(d) is str:
        return parse(d)
    else:
        return d

def startvalue(startdate):
    return startdate + delta(days=-25)

def enddate(startdate, rng):
    return startdate + delta(days=rng)

def TimeFormat(hrs=False, mins=False, secs=False):
    if secs:
        return '%Y-%m-%d %H:%M:%S'
    elif mins:
        return '%Y-%m-%d %H:%M'
    elif hrs:
        return '%Y-%m-%d %H'
    else:
        return '%Y-%m-%d'

def printTime(start):
    end = time()
    duration = end - start
    if duration < 60:
        ans = '{:.3f}s'.format(duration)
    else:
        mins = duration / 60
        secs = duration % 60
        ans = '{:.0f}m {:.3f}s'.format(mins, secs)
        
    print(ans)

def line():
    return '\n'

def dline():
    return line() + line()

def matrix(df, cols):
    return df.pivot(cols[0], cols[1], cols[2])

def heatmap(df, cols, title='', dims=(15,15)):
    import seaborn as sns
    from matplotlib import pyplot

    _, ax = pyplot.subplots(figsize=dims)
    ax.set_title(title)
    return sns.heatmap(ax=ax, data=matrix(df, cols), annot=True, annot_kws={"size":8}, fmt='.1f')

def readcsv(startdate, daterange, symbol=None):
    p = Path.cwd().parent / 'Testing/df.csv'
    df = pd.read_csv(p, parse_dates=['Timestamp'], index_col=0)

    if not symbol is None:
        df = filterdf(dfall=df, symbol=symbol)

    mask = ((df['Timestamp'] >= startvalue(startdate)) & (df['Timestamp'] <= enddate(startdate, daterange)))
    df = df.loc[mask].reset_index(drop=True)
    return df

def plotChart(df, symbol, df2=None):
    import plotly.offline as py
    py.iplot(chart(df=df, symbol=symbol, df2=df2))

def addorder(order, ts, price=None):    
    import plotly.graph_objs as go

    if price is None:
        price = order.price
        if order.marketfilled:
            linedash = 'dash'
        else:
            linedash = None
    else:
        linedash = 'dot'
    
    ft = order.filledtime
    if ft is None:
        ft = ts + delta(hours=24)
        linedash = 'dashdot'
    elif ft == ts:
        ft = ts + delta(hours=1)
    
    if not order.cancelled:
        if order.ordtype == 2:
            color = 'red'
        else:
            color = 'lime' if order.side == 1 else 'orange'
    else:
        color = 'grey'

    return go.layout.Shape(
        type='line',
        x0=ts,
        y0=price,
        x1=ft,
        y1=price,
        line=dict(
            color=color,
            width=2,
            dash=linedash))

def chartorders(df, t, pre=36, post=36, width=900, fast=50, slow=200):
    import plotly.graph_objs as go
    import plotly.offline as py

    ts = t.candles[0].Timestamp
    timelower = ts - delta(hours=pre)
    timeupper = ts + delta(hours=post)
    
    mask = (df['Timestamp'] >= timelower) & (df['Timestamp'] <= timeupper)
    df = df.loc[mask].reset_index(drop=True)
    
    shapes, x, y, text = [], [], [], []
    for order in t.allorders():
        shapes.append(addorder(order, ts=ts))
        if order.marketfilled or order.cancelled:
            shapes.append(addorder(order, ts=ts, price=order.pxoriginal))
        
        x.append(ts + delta(hours=-1))
        y.append(order.price),
        text.append('({:,}) {}'.format(order.contracts, order.name[:2]))

    labels = go.Scatter(x=x, y=y, text=text, mode='text', textposition='middle left')

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[f'ema{fast}'],line=dict(color='#18f27d', width=1)))
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[f'ema{slow}'],line=dict(color='#9d19fc', width=1)))
    fig.add_trace(candlestick(df))
    fig.add_trace(labels)
    
    scale = 0.8

    fig.update_layout(
        height=750 * scale,
        width=width * scale,
        paper_bgcolor='#000B15',
        plot_bgcolor='#000B15',
        font_color='white',
        showlegend=False,
        title='Trade {}: {}, {:.2%}, {}'.format(t.tradenum, t.conf, t.pnlfinal, t.duration()),
        shapes=shapes,
        xaxis_rangeslider_visible=False,
        yaxis=dict(side='right',
			showgrid=False,
			autorange=True,
			fixedrange=False) ,
        yaxis2=dict(showgrid=False,
            overlaying='y',
            autorange=False,
            fixedrange=True,
            range=[0,0.4])
            )
    fig.update_xaxes(
        autorange=True,
        tickformat=TimeFormat(hrs=True),
        gridcolor='#e6e6e6',
        showgrid=False,
        gridwidth=1,
        tickangle=315)
    
    fig.show()

def candlestick(df):
    import plotly.graph_objs as go
    trace = go.Candlestick(
        x=df['Timestamp'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing=dict(line=dict(color='#6df7ff')),
        decreasing=dict(line=dict(color='#ff6d6d')),
        line=dict(width=1),
        hoverlabel=dict(split=False, bgcolor='rgba(0,0,0,0)'))
    return trace

def chart(df, symbol, df2=None):
    import plotly.graph_objs as go
    import plotly.offline as py
    from plotly.subplots import make_subplots

    py.init_notebook_mode(connected=False)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_width=[0.1, 0.1, 0.4])

    fig.append_trace(candlestick(df), row=1, col=1)
    
    # add ema cols 
    for col in df.columns:
        if 'ema' in col:
            if '50' in col: sColour = '#18f27d'
            if '200' in col: sColour = '#9d19fc'
            emaTrace = go.Scatter(
                x=df['Timestamp'],
                y=df[col],
                mode='lines',
                line=dict(color=sColour, width=1))
            fig.append_trace(emaTrace, row=1, col=1)

    df3 = df2.loc[df2['PercentChange'] < 0]
    percentTrace = go.Bar(x=df3['Timestamp'], 
               y=df3['PercentChange'], 
               marker=dict(line=dict(width=2, color='#ff7a7a')))
    fig.append_trace(percentTrace, row=2, col=1)
    
    df3 = df2.loc[df2['PercentChange'] >= 0]
    percentTrace = go.Bar(x=df3['Timestamp'], 
               y=df3['PercentChange'], 
               marker=dict(line=dict(width=2, color='#91ffff')))
    fig.append_trace(percentTrace, row=2, col=1)

    balanceTrace = go.Scatter(
                    x=df2['Timestamp'],
                    y=df2['Balance'],
                    mode='lines',
                    line=dict(color='#91ffff', width=1))
    fig.append_trace(balanceTrace, row=3, col=1)

    fig.update_layout(
        height=800,
        paper_bgcolor='#000B15',
        plot_bgcolor='#000B15',
        font_color='white',
        showlegend=False,
        title=symbol + ' OHLC')
    fig.update_xaxes(
        autorange=True,
        tickformat=TimeFormat(hrs=True),
        gridcolor='#e6e6e6',
        showgrid=True,
        gridwidth=1,
        tickangle=315)
    fig.update_yaxes(
        autorange=True,
        fixedrange=False,
        side='right',
        row=1, col=1)
    fig.update_yaxes(
        showgrid=False,
        autorange=True,
        fixedrange=False,
        side='right',
        row=2, col=1)
    fig.update_yaxes(
        showgrid=False,
        autorange=True,
        fixedrange=False,
        side='right',
        row=3, col=1)

    return fig
 
def percent(val):
    return '{:.2%}'.format(val)

def priceformat(altstatus=False):
    ans = '{:,.0f}' if not altstatus else '{:,.0f}'
    return ans

def getPrice(pnl, entryprice, side):
    if side == 1:
        return pnl * entryprice + entryprice
    elif side == -1:
        return entryprice / (1 + pnl)

def getPnlXBT(contracts=0, entryprice=0, exitprice=0, isaltcoin=False):
    if 0 in (entryprice, exitprice):
        return 0
    elif not isaltcoin:
        return round(contracts * (1 / entryprice - 1 / exitprice), 8)
    elif isaltcoin:
        return round(contracts * (exitprice - entryprice), 8)

def getPnl(side, entryprice, exitprice):
    if 0 in (entryprice, exitprice):
        return 0
    elif side == 1:
        return round((exitprice - entryprice) / entryprice, 4)
    else:
        return round((entryprice - exitprice) / exitprice, 4)

def getContracts(xbt, leverage, entryprice, side, isaltcoin=False):
    if not isaltcoin:
        return int(xbt * leverage * entryprice * side)
    else:
        return int(xbt * leverage * (1 / entryprice) * side)

def side(x):
    side = lambda x: -1 if x < 0 else (1 if x > 0 else 0)
    return side(x)

def convert_bitmex(o):
    # Covert Bitmex manual orders to c.Order()
    # TODO: This doesn't preserve all attributes (eg orderID), but creates a new order from scratch.
    price = o['stopPx'] if o['ordType'] == 'Stop' else o['price']

    return c.Order(
        symbol=o['symbol'],
        price=price,
        contracts=o['contracts'],
        ordtype=o['ordType'],
        name=o['name'])

def usefulkeys(orders):
    # return only useful keys from bitmex orders in dict form
    keys = ('symbol', 'clOrdID', 'side', 'price', 'stopPx', 'ordType', 'execInst', 'ordStatus', 'contracts', 'name', 'manual', 'orderID')
    if not isinstance(orders, list):
        islist = False
        orders = [orders]
    else:
        islist = True

    result = [{key: o[key] for key in o.keys() if key in keys} for o in orders]
    
    if islist:
        return result
    else:
        return result[0]

def key(symbol, name, side, ordtype):
    # if ordtype == 'Stop':
    #     side *= -1
    
    sidestr = 'long' if side == 1 else 'short'

    return '{}-{}-{}'.format(symbol, name.lower(), sidestr)

def col(df, col):
    return df.columns.get_loc(col)

def discord(msg, channel='jambot'):
    import requests
    import discord
    from discord import Webhook, RequestsWebhookAdapter, File

    p = Path(topfolder) / 'data/ApiKeys/discord.csv'
    r = pd.read_csv(p, index_col='channel').loc[channel]
    if channel == 'err': msg += '@here' 

    # Create webhook
    webhook = Webhook.partial(r.id, r.token, adapter=RequestsWebhookAdapter())
    
    # split into strings of max 2000 char for discord
    n = 2000
    out = [(msg[i:i+n]) for i in range(0, len(msg), n)]
    
    for msg in out:
        webhook.send(msg)

def senderror(msg='', prnt=False):
    import traceback
    err = traceback.format_exc().replace('Traceback (most recent call last):\n', '')

    if not msg == '':
        err = '{}:\n{}'.format(msg, err).replace(':\nNoneType: None', '')
    
    err = '*------------------*\n{}'.format(err)

    if prnt or not 'linux' in platform:
        print(err)
    else:
        discord(msg=err, channel='err')
    # return err


# DATABASE
def getGoogleSheet():
    import pygsheets
    p = Path(topfolder) / 'data/ApiKeys/gsheets.json'
    return pygsheets.authorize(service_account_file=p).open('Jambot Settings')

def getDataFrame(symbol=None, period=300, startdate=None, enddate=None, daterange=None, interval=1, db=None, offset=-15):
    if db is None: db = DB()
    
    if startdate is None:
        startdate = timenow(interval=interval) + delta(hours=abs(period) * -1)
    else:
        if enddate is None and not daterange is None:
            enddate = startdate + delta(days=daterange)

        startdate += delta(days=offset)

    tbl = pk.Table('Bitmex')
    q = (pk.Query.from_(tbl)
        .select('Symbol', 'Timestamp', 'Open', 'High', 'Low', 'Close')
        .where(tbl.Interval==interval)
        .where(tbl.Timestamp>=startdate)
        .orderby('Symbol', 'Timestamp'))

    if not symbol is None: q = q.where(tbl.Symbol==symbol)
    if not enddate is None: q = q.where(tbl.Timestamp<=enddate)
    
    df = pd.read_sql_query(sql=q.get_sql(), con=db.conn, parse_dates=['Timestamp'])
    db.close()

    return df

def getmaxdates(db=None, interval=1):
    # Make sure to close conn
    if db is None: db = DB()

    tbl = pk.Table('Bitmex')
    q = (pk.Query.from_(tbl)
        .select('Interval', 'Symbol', fn.Max(tbl.Timestamp)
            .as_('Timestamp'))
        .where(tbl.Interval==interval)
        .groupby(tbl.Symbol, tbl.Interval)
        .orderby('Timestamp'))

    df = pd.read_sql_query(sql=q.get_sql(), con=db.conn, parse_dates=['Timestamp'])
    df.Interval = df.Interval.astype('int')

    return df

def getInterval(interval='1h'):
    if interval == '1h':
        return 1
    elif interval == '15min':
        return 15    

def getDelta(interval=1):
    if interval == 1:
        return delta(hours=1)
    elif interval == 15:
        return delta(minutes=15)

def timenow(interval=1):
    if interval == 1:
        return date.utcnow().replace(microsecond=0, second=0, minute=0)
    elif interval == 15:
        return round_minutes(dt=date.utcnow(), resolution=15).replace(microsecond=0, second=0)

def round_minutes(dt, resolution):
    new_minute = (dt.minute // resolution) * resolution
    return dt + delta(minutes=new_minute - dt.minute)

def updateAllSymbols(u=None, db=None, interval=1):
    lst = []
    if u is None: u = live.User()
    if db is None: db = DB()

    # loop query result, add all to dict with maxtime as KEY, symbols as LIST
    m = defaultdict(list)
    for _, row in getmaxdates(db, interval=interval).iterrows():
        m[row.Timestamp].append(row.Symbol)

    # loop dict and call bitmex for each list of syms in maxdate
    for maxdate in m.keys():
        starttime = maxdate + getDelta(interval)
        if starttime < timenow(interval):
            fltr = json.dumps(dict(symbol=m[maxdate])) # filter symbols needed

            dfcandles = u.getCandles(starttime=starttime, fltr=fltr, includepartial=False, interval=interval)
            lst.append(dfcandles)

    if lst:
        df = pd.concat(lst) # maybe remove duplicates
        df.to_sql(name='Bitmex', con=db.conn, if_exists='append', index=False)
    
    # TODO: keep db conn open and pass to TopLoop
    db.close()

def qIt(s):
    return "'" + s + "'"

def get_db():
    p = Path(topfolder) / 'data/ApiKeys/db.yaml'
    with open(p) as file:
        m = yaml.full_load(file)
    return m

def strConn():
    m = get_db()
    return ';'.join('{}={}'.format(k, v) for k, v in m.items())

def engine():
    print('loading db')
    params = prse.quote_plus(strConn())
    return sa.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(params), fast_executemany=True)

# TODO: move database into its own class
class DB(object):
    def __init__(self):
        self.df_unit = None
        self.conn = engine()
        self.conn.raw_connection().autocommit = True  # doesn't seem to work rn
        self.cursor = self.conn.raw_connection().cursor()

    def close(self):
        try:
            self.cursor.close()
        except:
            try:
                self.conn.raw_connection().close()
            except:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Pandas DataFrame Styles
def background_contracts(s):
    # color = 'red' if val < 0 else 'black'
    # return 'color: {}'.format(color)
    is_neg = s < 0
    return ['background-color: red' if v else 'background-color: blue' for v in is_neg]

def neg_red(s):
    return ['color: #ff8080' if v < 0 else '' for v in s]

# Pandas plot on 2 axis
# ax = sym.df.plot(kind='line', x='Timestamp', y=['ema50', 'ema200'])
# sym.df.plot(kind='line', x='Timestamp', y='conf', secondary_y=True, ax=ax)


# Column math functions
def addEma(df, p, c='Close'):
    # check if column already exists
    col = f'ema{p}'
    if not col in df.columns:
        df[col] = df[c].ewm(span=p, min_periods=p).mean()

class Switch:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False # Allows a traceback to occur

    def __call__(self, *values):
        return self.value in values
