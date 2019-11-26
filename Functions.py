# General Functions module
import json
import os
from collections import defaultdict
from datetime import datetime as date
from datetime import timedelta as delta
from sys import platform
from time import time
from urllib import parse as prse

import pandas as pd
import pypika as pk
import sqlalchemy as sa
from dateutil.parser import parse
from pypika import functions as fn

import LiveTrading as live
import pyodbc


# PARALLEL
def runtrend(symbol, startdate, mr, df, against, wth, row, titles):
    import JambotClasses as c
    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])

    # Strat_Trend
    trend = c.Strat_Trend(speed=(against, wth), mr=mr)
    strats = []
    strats.append(trend)

    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=strats, df=df, row=row)
    sym.decidefull()

    a = sym.account
    dfTemp.loc[0] = [against, wth, round(a.min,3), round(a.max,3), round(a.balance,3), sym.strats[0].tradecount()]

    return dfTemp

def runtrendrev_single(startdate, dfall, speed, norm, row):
    import JambotClasses as c

    symbol = row.symbol
    df = dfall[dfall.Symbol==symbol].reset_index(drop=True)

    strat = c.Strat_TrendRev(speed=speed, norm=norm)
    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=[strat], df=df, row=row)
    sym.decidefull()

    return sym

def runtrendrev(symbol, startdate, df, against, wth, row, titles, norm):
    import JambotClasses as c
    

    # Strat_TrendRev
    strat = c.Strat_TrendRev(speed=(against, wth), norm=norm)
    sym = c.Backtest(symbol=symbol, startdate=startdate, strats=[strat], df=df, row=row)
    sym.decidefull()

    a = sym.account

    dfTemp = pd.DataFrame(columns=[titles[0], titles[1], 'min', 'max', 'final', 'numtrades'])
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

def hournow():
    return date.utcnow().replace(microsecond=0, second=0, minute=0)


def line():
    return '\n'

def dline():
    return line() + line()

def matrix(df, cols):
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
        if order.ordtype == 2:
            color = 'red'
        else:
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
    # fig.add_trace(go.Scatter(x=df['CloseTime'], y=df['ema10'],line=dict(color='#ffb066', width=1)))
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

def priceformat(altstatus=False):
    ans = '{:,.0f}' if not altstatus else '{:,.0f}'
    return ans

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
    # x = int(x)
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

        msg += '@here'

    # Create webhook
    webhook = Webhook.partial(WEBHOOK_ID, WEBHOOK_TOKEN,\
    adapter=RequestsWebhookAdapter())
    
    if len(msg) > 0:
        webhook.send(msg)

def senderror(msg='', prnt=False):
    import traceback
    err = traceback.format_exc().replace('Traceback (most recent call last):\n', '')

    if not msg == '':
        err = '{}:\n{}'.format(msg, err).replace(':\nNoneType: None', '')
    
    err = '*------------------*\n{}'.format(err)

    if prnt or not 'linux' in platform:
        print(err)
        discord(err, channel='err')
    else:
        discord(err, channel='err')


# DATABASE
def getGoogleSheet():
    import pygsheets
    fpath = os.path.join(currentdir(), 'client_secret.json')
    c = pygsheets.authorize(service_account_file=fpath)
    sheet = c.open("Jambot Settings")
    return sheet

def getDataFrame(symbol=None, period=300, startdate=None, enddate=None):
    e = engine()

    sql=getQuery(symbol=symbol, period=period, startdate=startdate, enddate=enddate)
    Query = pd.read_sql_query(sql=sql, con=e, parse_dates=['CloseTime'])
    e.raw_connection().close()

    return pd.DataFrame(Query)

def getmaxdates(db=None):
    # Make sure to close conn
    if db is None: db = DB()

    tbl = pk.Table('Bitmex_OHLC')
    q = (pk.Query.from_(tbl)
        .select('Symbol', fn.Max(tbl.CloseTime)
            .as_('CloseTime'))
        .groupby(tbl.Symbol)
        .orderby('CloseTime'))

    df = pd.read_sql_query(sql=q.get_sql(), con=db.conn, parse_dates=['CloseTime'])

    return df

def updateAllSymbols(u=None):
    db = DB()
    lst = []
    if u is None: u = live.User()
    datenow = hournow()

    # loop query result, add all to dict with maxtime as KEY, symbols as LIST
    m = defaultdict(list)
    for i, row in getmaxdates(db).iterrows():
        m[row.CloseTime].append(row.Symbol)

    # loop dict and call bitmex for each list of syms in maxdate
    for maxdate in m.keys():
        fltr = json.dumps(dict(symbol=m[maxdate])) # filter symbols needed
        numsyms = len(m[maxdate])

        while maxdate < datenow:
            dfcandles = u.getCandles(starttime=maxdate + delta(hours=1), f=fltr, includepartial=False)
            lst.append(dfcandles)

            maxdate += delta(hours=1000 // numsyms)

    if lst:
        df = pd.concat(lst)
        df.to_sql(name='Bitmex_OHLC', con=db.conn, if_exists='append', index=False)
    
    # TODO: keep db conn open and pass to TopLoop
    db.close()

def getQuery(symbol=None, period=300, startdate=None, enddate=None):

    # TODO: Rebuild this with pypika
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

def strConn():
    driver = '{ODBC Driver 17 for SQL Server}'
    server = 'tcp:jgazure2.database.windows.net,1433'
    database = 'Jambot'
    username = 'jgordon@jgazure2'
    password = 'Z%^7wdpf%Nai=^ZFy-U.'
    return 'DRIVER={};SERVER={};DATABASE={};UID={};PWD={}'.format(driver, server, database, username, password)

def engine():
    params = prse.quote_plus(strConn())
    return sa.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(params), fast_executemany=True)

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
# ax = sym.df.plot(kind='line', x='CloseTime', y=['ema50', 'ema200'])
# sym.df.plot(kind='line', x='CloseTime', y='conf', secondary_y=True, ax=ax)


# Column math functions
def addEma(df, p, c='Close'):
    # check if column already exists
    col = 'ema{}'.format(p)
    if not col in df:
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


