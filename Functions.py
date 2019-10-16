# General Functions module
import json
import os
from collections import defaultdict
from datetime import datetime as date
from datetime import timedelta as delta
from sys import platform
from time import time

import numpy as np
import pandas as pd
from dateutil.parser import parse

import pyodbc


def hurst(ts):
    from numpy import cumsum, log, polyfit, sqrt, std, subtract
    from numpy.random import randn

    # Create the range of lag values
    lags = range(2, 100)

	# Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def runtrend(symbol, startdate, mr, df, against, wth, row, titles, opposite):
    import JambotClasses as c
    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])

    # Strat_Trend
    trend = c.Strat_Trend(speed=(against, wth), mr=mr, opposite=opposite)
    strats = []
    strats.append(trend)

    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    sym.decidefull()

    a = sym.account
    dfTemp.loc[0] = [against, wth, round(a.min,3), round(a.max,3), round(a.balance,3), sym.strats[0].tradecount()]

    return dfTemp

def runtrendrev(symbol, startdate, df, against, wth, row, titles):
    import JambotClasses as c
    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])

    # Strat_Trend
    trend = c.Strat_TrendRev(speed=(against, wth))
    strats = []
    strats.append(trend)

    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    sym.decidefull()

    a = sym.account
    dfTemp.loc[0] = [against, wth, round(a.min,3), round(a.max,3), round(a.balance,3), sym.strats[0].tradecount()]

    return dfTemp

def runchop(symbol, startdate, df, against, wth, tpagainst, tpwith, lowernorm, uppernorm, row, titles):
    import JambotClasses as c
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

def line():
    return '\n'

def dline():
    return line() + line()

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
        return "used: " + str(round(duration, 2)) + "s."
    else:
        mins = int(duration / 60)
        secs = round(duration % 60, 2)
        if mins < 60:
            return "used: " + str(mins) + "m " + str(secs) + "s."
        else:
            hours = int(duration / 3600)
            mins = mins % 60
            return "used: " + str(hours) + "h " + str(mins) + "m " + str(secs) + "s."

def matrix(df, cols):
    # print(df)
    return df.pivot(cols[0], cols[1], cols[2])

def heatmap(df, cols, title='', dims=(15,15)):
    import seaborn as sns
    from matplotlib import pyplot

    fig, ax = pyplot.subplots(figsize=dims)
    ax.set_title(title)
    return sns.heatmap(ax=ax, data=matrix(df, cols), annot=True, annot_kws={"size":8}, fmt='.1f')

def readcsv(startdate, daterange):
	df = pd.read_csv('df.csv', parse_dates=['CloseTime'], index_col=0)
	mask = (df['CloseTime'] >= startvalue(startdate)) & (df['CloseTime'] <= enddate(startdate, daterange))
	df = df.loc[mask].reset_index(drop=True)
	return df

def currentdir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '')

def getGoogleSheet():
    import pygsheets
    fpath = os.path.join(currentdir(), 'client_secret.json')
    c = pygsheets.authorize(service_account_file=fpath)
    sheet = c.open("Jambot Settings")
    return sheet

def getConn():
    server = 'tcp:jgazure2.database.windows.net,1433'
    database = 'Jambot'
    username = 'jgordon@jgazure2'
    password = 'Z%^7wdpf%Nai=^ZFy-U.'
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' +
                          server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    return conn

def getDataFrame(symbol=None, period=300, startdate=None, enddate=None):
    conn = getConn()
    sql=getQuery(symbol=symbol, period=period, startdate=startdate, enddate=enddate)
    Query = pd.read_sql_query(sql=sql, con=conn, parse_dates=['CloseTime'])
    conn.close

    return pd.DataFrame(Query)

def updateAllSymbols():
    import LiveTrading as live
    sql = 'Select Symbol, DateAdd(hour, 1, max(CloseTime)) as CloseTime From Bitmex_OHLC Group By Symbol Order By CloseTime'
    conn = getConn()
    df = pd.read_sql_query(sql=sql, con=conn, parse_dates=['CloseTime'])
    result = None

    # loop query result, add all to dict with maxtime as key, symbols as list
    m = defaultdict(list)
    for i, row in df.iterrows():
        m[row.CloseTime].append(row.Symbol)
    # print(m)
    u = live.User() # could maybe get this from a global variable, or pass in?
    sql = 'Insert Into Bitmex_OHLC (Symbol, Interval, CloseTime, [Open], High, Low, [Close]) Values '

    # loop dict and call bitmex for each group of syms in max time
    for starttime in m.keys():

        numsyms = len(m[starttime])
        hrsremaining = 1 + (date.utcnow() - starttime).total_seconds() // 3600
        # print(date.utcnow(), starttime, hrsremaining)
        
        while hrsremaining > 0:
            f = json.dumps(dict(symbol=m[starttime]))
            result = u.getCandles(startTime=starttime, f=f, reverse=False)
            # print(json.dumps(result, default=str, indent=4))

            for c in result:
                sql += '({}, 1, {}, {}, {}, {}, {}),'.format(qIt(c['symbol']), qIt(date.strftime(c['timestamp'], TimeFormat(secs=True))), c['open'], c['high'], c['low'], c['close'])

            starttime += delta(hours=750 // numsyms)
            hrsremaining = (date.now() - starttime).total_seconds() // 3600
    
    if not result is None and len(result) > 0:
        # print(sql[:-1])
        conn.cursor().execute(sql[:-1])
        conn.commit()
    
    conn.close
    # return result

def getQuery(symbol=None, period=300, startdate=None, enddate=None):

    if startdate is None:
        startdate = date.utcnow() + delta(hours=abs(period) * -1)
        # startdate = t.strftime(TimeFormat())

    sFilter = 'Where CloseTime>=' + qIt(startdate.strftime(TimeFormat(mins=True)))

    if not enddate is None:
        sFilter = sFilter + ' and CloseTime<=' + qIt(enddate.strftime(TimeFormat(mins=True)))

    if not symbol is None:
        sFilter = sFilter + ' and Symbol=' + qIt(symbol)

    Query = 'Select Symbol, CloseTime, [Open], High, Low, [Close] From Bitmex_OHLC ' + \
        sFilter + ' Order By Symbol, CloseTime '
    # print(Query)
    return Query

def qIt(s):
    return "'" + s + "'"

def plotChart(df, symbol, df2=None):
    import plotly.offline as py
    py.iplot(chart(df=df, symbol=symbol, df2=df2))

def addorder(order, ts):    
    import plotly.graph_objs as go
    
    ft = order.filledtime
    if ft is None:
        ft = ts + delta(hours=24)
    elif ft == ts:
        ft = ts + delta(hours=1)
    
    if not order.cancelled:
        color = 'lime' if order.side == 1 else 'orange'
    else:
        color = 'grey'

    return go.layout.Shape(
        type='line',
        x0=ts,
        y0=order.price,
        x1=ft,
        y1=order.price,
        line=dict(
            color=color,
            width=2))

def chartorders(df, t, n=36):
    import plotly.graph_objs as go
    import plotly.offline as py

    # ts = date(2019, 8, 26)
    ts = t.candles[0].CloseTime
    timelower = ts - delta(hours=n)
    timeupper = ts + delta(hours=n)

    mask = (df['CloseTime'] >= timelower) & (df['CloseTime'] <= timeupper)
    df = df.loc[mask].reset_index(drop=True)
    
    shapes, x, y, text = [], [], [], []
    for order in t.allorders():
        shapes.append(addorder(order, ts=ts))
        x.append(ts + delta(hours=-1))
        y.append(order.price),
        text.append('({}) {}'.format(order.contracts, order.name[:2]))

    labels = go.Scatter(x=x, y=y, text=text, mode='text', textposition='middle left')

    fig = go.Figure()
    
    # fig.add_trace(go.Scatter(x=df['CloseTime'], y=df['norm'],line=dict(color='yellow', width=1), yaxis='y2'))
    # fig.add_trace(go.Scatter(x=df['CloseTime'], y=df['trendrev_high'],line=dict(color='red', width=1)))
    # fig.add_trace(go.Scatter(x=df['CloseTime'], y=df['trendrev_low'],line=dict(color='red', width=1)))
    fig.add_trace(go.Scatter(x=df['CloseTime'], y=df['ema10'],line=dict(color='#ffb066', width=1)))
    fig.add_trace(go.Scatter(x=df['CloseTime'], y=df['ema50'],line=dict(color='#18f27d', width=1)))
    fig.add_trace(go.Scatter(x=df['CloseTime'], y=df['ema200'],line=dict(color='#9d19fc', width=1)))
    fig.add_trace(candlestick(df))
    fig.add_trace(labels)
    
    scale = 0.8

    fig.update_layout(
        height=750 * scale,
        width=900 * scale,
        paper_bgcolor='#000B15',
        plot_bgcolor='#000B15',
        font_color='white',
        showlegend=False,
        title='Trade {}: {}, {:.2%}, {}'.format(t.tradenum, t.conf, t.pnlfinal, t.duration()),
        shapes=shapes,
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
        showgrid=True,
        gridwidth=1,
        tickangle=315)
    
    fig.show()

def candlestick(df):
    import plotly.graph_objs as go
    trace = go.Candlestick(
        x=df['CloseTime'],
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
                x=df['CloseTime'],
                y=df[col],
                mode='lines',
                line=dict(color=sColour, width=1))
            fig.append_trace(emaTrace, row=1, col=1)

    df3 = df2.loc[df2['PercentChange'] < 0]
    percentTrace = go.Bar(x=df3['CloseTime'], 
               y=df3['PercentChange'], 
               marker=dict(line=dict(width=2, color='#ff7a7a')))
    fig.append_trace(percentTrace, row=2, col=1)
    
    df3 = df2.loc[df2['PercentChange'] >= 0]
    percentTrace = go.Bar(x=df3['CloseTime'], 
               y=df3['PercentChange'], 
               marker=dict(line=dict(width=2, color='#91ffff')))
    fig.append_trace(percentTrace, row=2, col=1)

    balanceTrace = go.Scatter(
                    x=df2['CloseTime'],
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

def getPrice(pnl, entryprice, side):
    if side == 1:
        return pnl * entryprice + entryprice
    elif side == -1:
        return entryprice / (1 + pnl)

def getPnlXBT(contracts, entryprice, exitprice, isaltcoin=False):
    if not isaltcoin:
        return round(contracts * (1 / entryprice - 1 / exitprice), 8)
    elif isaltcoin:
        return round(contracts * (exitprice - entryprice), 8)

def getPnl(side, entryprice, exitprice):
    if entryprice == 0 or exitprice == 0:
        return 0
    elif side == 1:
        return round((exitprice - entryprice) / entryprice,4)
    else:
        return round((entryprice - exitprice) / exitprice, 4)

def getContracts(xbt, leverage, entryprice, side, isaltcoin=False):
    if not isaltcoin:
        return int(xbt * leverage * entryprice * side)
    else:
        return int(xbt * leverage * (1 / entryprice) * side)

def side(x):
    x = int(x)
    side = lambda x: -1 if x < 0 else (1 if x > 0 else 0)
    return side(x)

def sidestr(side, ordtype):
    # opposite only for stops
    if not ordtype == 1:
        side *= -1
    return 'long' if side == 1 else 'short'

def key(symbol, name, side, ordtype):
    return '{}-{}-{}'.format(symbol, name.lower(), sidestr(side, ordtype))

def col(df, col):
    return df.columns.get_loc(col)

def discord(msg, channel='jambot'):
    import requests
    import discord
    from discord import Webhook, RequestsWebhookAdapter, File

    if channel == 'orders':
        WEBHOOK_ID = '546699038704140329'
        WEBHOOK_TOKEN = '95ZgQfMEv7sj8qGGUciuaMmjHJuxpskG-0nYjOCSZCGBnnSr93MBj7j9_R7nfC1f3AIC'
    elif channel == 'jambot':
        WEBHOOK_ID = '506472880620568576'
        WEBHOOK_TOKEN = 'o1EkqIpGizc5ewUyjivFeEAkvgbU_91qr6Pi-FDLP0qCzu-j7yNFc9vskULJ53JZ6aC1'
    elif channel == 'err':
        WEBHOOK_ID = '512030769775116319'
        WEBHOOK_TOKEN = 's746HqzlZGedOfnSmgDeC8HJJT_5-bYcgUbgs8KWwvb6gw38gGR_WhQylFKdcWtGyTHi'

    # Create webhook
    webhook = Webhook.partial(WEBHOOK_ID, WEBHOOK_TOKEN,\
    adapter=RequestsWebhookAdapter())
    
    if len(msg) > 0:
        webhook.send(msg)

def senderror(msg='',prnt=False):
    import traceback
    err = traceback.format_exc().replace('Traceback (most recent call last):\n', '')
    
    if not msg == '':
        err = '{}:\n{}'.format(msg, err)
    
    err = '*------------------*\n{}'.format(err)

    if prnt or not 'linux' in platform:
        print(err)
    else:
        discord(err)

# Column math functions

def getC(maxspread):
    m = -2.9
    b = 0.135
    return round(m * maxspread + b, 2)

def emaExp(x, c):
    side = np.where(x >= 0, 1, -1)
    x = abs(x)
    
    aLim = 2
    a = -1000
    b = 3
    d = -3
    g = 1.7

    y = side * (a * x ** b + d * x ** g) / (aLim * a * x ** b + aLim * d * x ** g + c)

    return round(y, 6)

def setConf(df):
    
    return df

def addEma(df, p, c='Close'):
    df['ema{}'.format(p)] = df[c].ewm(span=p, min_periods=p).mean()

def setTradePrices(name, df, speed):

    addEma(df=df, p=10)
    addEma(df=df, p=50)
    addEma(df=df, p=200)

    df['emaspread'] = round((df['ema50'] - df['ema200']) / ((df['ema50'] + df['ema200']) / 2) ,6)
    df['trend'] = np.where(df['ema50'] > df['ema200'], 1, -1)

    c = getC(maxspread=0.1)
    df['conf'] = emaExp(x=df['emaspread'], c=c)
    
    against, wth = speed[0], speed[1]
    df['mhw'] = df['High'].rolling(wth).max().shift(1)
    df['mha'] = df['High'].rolling(against).max().shift(1)
    df['mla'] = df['Low'].rolling(wth).min().shift(1)
    df['mlw'] = df['Low'].rolling(against).min().shift(1)

    df[name + '_high'] = np.where(df['trend'] == 1, df['mha'], df['mhw'])
    df[name + '_low'] = np.where(df['trend'] == -1, df['mlw'], df['mla'])
    
    df = df.drop(columns=['mha', 'mhw', 'mla', 'mlw'])

    return df

def setVolatility(df, norm=(1,4)):
    df['maxhigh'] = df['High'].rolling(48).max()
    df['minlow'] = df['Low'].rolling(48).min()
    df['spread'] = abs(df['maxhigh'] - df['minlow']) / df[['maxhigh', 'minlow']].mean(axis=1)

    df['emavty'] = df['spread'].ewm(span=180, min_periods=180).mean()
    df['smavty'] = df['spread'].rolling(300).mean()
    df['norm'] = np.interp(df['smavty'], (0, 0.25), (norm[0], norm[1]))
    df['normtp'] = np.interp(df['smavty'], (0, 0.4), (0.3, 3))

    df = df.drop(columns=['maxhigh', 'minlow'])

    return df


class Switch:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False # Allows a traceback to occur

    def __call__(self, *values):
        return self.value in values

# Pandas plot on 2 axis
# ax = sym.df.plot(kind='line', x='CloseTime', y=['ema50', 'ema200'])
# sym.df.plot(kind='line', x='CloseTime', y='conf', secondary_y=True, ax=ax)