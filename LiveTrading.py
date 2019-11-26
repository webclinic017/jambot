import json
import os
import time
from collections import defaultdict
from datetime import datetime as date
from datetime import timedelta as delta

import pandas as pd
import numpy as np
from bitmex import bitmex

import Functions as f
import JambotClasses as c


class User():
    def __init__(self, test=False):
        
        self.name = ''
        self.nameshort = ''
        self.percentbalance = 1
        self.gRow = 1
        self.availablemargin = 0
        self.totalbalancemargin = 0
        self.totalbalancewallet = 0
        self.reservedbalance = 0
        self.unrealizedpnl = 0
        self.orders = None
        self.positions = None
        self.partialcandle = None
        self.div = 100000000

        df = pd.read_csv(os.path.join(f.currentdir(), 'api.csv'))
        user = 'jayme' if not test else 'testnet'
        api_key = df.apikey.loc[df.user == user].values[0]
        api_secret = df.apisecret.loc[df.user == user].values[0]

        self.client = bitmex(test=test, api_key=api_key, api_secret=api_secret)
        self.setTotalBalance()

    def balance(self):
        return self.totalbalancewallet - self.reservedbalance

    def checkrequest(self, request, retries=0):
        # request = bravado.http_future.HttpFuture
        # response = bravado.response.BravadoResponse
        response = request.response()
        status = response.metadata.status_code
        backoff = 0.5

        if status < 300:
            return response.result
        elif status == 503 and retries < 9:
            retries += 1
            sleeptime = backoff * (2 ** retries - 1)
            time.sleep(sleeptime)
            return self.checkrequest(request=request, retries=retries)
        else:
            # if data is blank, may need to .prepare() request first?
            f.senderror('{}\n{}'.format(response, request.future.request.data))

    def getPosition(self, symbol=None, refresh=False):
        if self.positions is None or refresh:
            self.setPositions()
        
        if not symbol is None:
            if symbol in self.posdict:
                return self.positions[self.posdict[symbol]]
            else:
                return {}
        else:
            return self.positions

    def setPositions(self, fltr=''):
        self.posdict = defaultdict()
        self.positions = self.client.Position.Position_get(filter=fltr).result()[0]
        
        for i, pos in enumerate(self.positions):
            self.posdict[pos['symbol']] = i

    def dfOrders(self, symbol='', newonly=True, refresh=False):
        orders = self.getOrders(symbol=symbol, newonly=newonly, refresh=refresh)
        cols = ['ordType', 'name', 'size', 'price', 'execInst', 'symbol']
        
        if not orders:
            df = pd.DataFrame(columns=cols, index=range(1))
        else:
            df = pd.io.json.json_normalize(orders)
            df['size'] = df.orderQty * df.side
            df['price'] = np.where(df.price > 0, df.price, df.stopPx)
        
        df = df.reindex(columns=cols).sort_values(
            by=['symbol', 'ordType', 'name'], 
            ascending=[False, True, True]).reset_index(drop=True)
        return df

    def getOrderByKey(self, key):
        if self.orders is None:
            self.setOrders(newonly=True)
        
        if key in self.orderkeysdict:
            return self.orders[self.orderkeysdict[key]]       
        else:
            return defaultdict(type(None))

    def getFilledOrders(self, symbol='', starttime=None):

        if starttime is None:
            starttime = date.utcnow() + delta(days=-7)

        fltr = json.dumps(dict(ordStatus='Filled'))

        # TODO: return this as df
        return self.client.Order.Order_getOrders(symbol=symbol,
                                                filter=fltr,
                                                count=100,
                                                startTime=starttime).response().result
        
    def getOrders(self, symbol='', newonly=True, refresh=False):
        if self.orders is None or refresh:
            self.orders = []
            self.setOrders(newonly=newonly)
        
        if symbol == '':
            return self.orders
        else:
            return [self.orders[i] for i in self.orderdict[symbol]]
            
    def setOrders(self, symbol='', fltr={}, newonly=True):
        self.orderdict = defaultdict(list)
        self.orderkeysdict = {}
        
        if newonly:
            fltr['ordStatus'] = 'New'

        fltr = json.dumps(fltr)

        self.orders = self.client.Order.Order_getOrders(symbol=symbol, filter=fltr, reverse=True, count=12).response().result
        for i, order in enumerate(self.orders):
            
            order['side'] = 1 if order['side'] == 'Buy' else -1

            # add key to the order
            if not order['clOrdID'] == '':
                order['key'] = '-'.join(order['clOrdID'].split('-')[:-1])
                order['name'] = '-'.join(order['clOrdID'].split('-')[1:-1])
                self.orderkeysdict[order['key']] = i

            self.orderdict[order['symbol']].append(i)

    def setTotalBalance(self):
        res = self.client.User.User_getMargin(currency='XBt').response().result
        self.availablemargin = res['excessMargin'] / self.div # total available/unused > only used in postOrder
        self.totalbalancemargin = res['marginBalance'] / self.div # unrealized + realized > don't actually use 
        self.totalbalancewallet = res['walletBalance'] / self.div # realized
        self.unrealizedpnl = res['unrealisedPnl'] / self.div
        # return res

    def amendbulk(self, amendorders):
        # accept list of Jambot.Order() objects, convert and send amend command to bitmex.
        try:
            orders = [order.amendorder() for order in amendorders]
            if not orders: return
            return self.checkrequest(self.client.Order.Order_amendBulk(orders=json.dumps(orders)))
        except:
            msg = ''
            for order in amendorders:
                msg += json.dumps(order.amendorder()) + '\n'
            f.senderror(msg)

    def placesingle(self, order):

        return self.client.Order.Order_new()

    def placebulk(self, placeorders):
        orders = []
        l = len(placeorders)

        for order in placeorders:
            if l > 1 and order.neworder().get('ordType', '') == 'Market':
                self.placebulk([order])
            else:
                orders.append(order.neworder())
                    
        if not orders: return
        try:
            return self.checkrequest(self.client.Order.Order_newBulk(orders=json.dumps(orders)))
        except:
            msg = ''
            for order in placeorders:
                msg += json.dumps(order.neworder()) + '\n'
            f.senderror(msg) 
    
    def cancelbulk(self, cancelorders):
        # only need ordID to cancel
        orders = [order['orderID'] for order in cancelorders]
        if not orders: return
        return self.checkrequest(self.client.Order.Order_cancel(orderID=json.dumps(orders)))
        
    def getpartial(self, symbol):
        
        # need to compare timestamp of partial
        
        if self.partialcandle is None or not self.partialcandle.Symbol[0] == symbol:
            self.setpartial(symbol=symbol)
        
        return self.partialcandle
        
    def setpartial(self, symbol):
        # call only partial candle from bitmex, save to self.partialcandle
        starttime = date.utcnow() + delta(hours=-1)
        self.getCandles(symbol=symbol, starttime=starttime)
    
    def appendpartial(self, df):
        # partial not built to work with multiple symbols, need to add partials to dict
        # Append partialcandle df to df from SQL db

        symbol = df.Symbol[0]
        dfpartial = self.getpartial(symbol=symbol)

        return df.append(dfpartial, sort=False).reset_index(drop=True)

    def getCandles(self, symbol='', starttime=None, f='', retainpartial=False, includepartial=True):
        result = self.client.Trade.Trade_getBucketed(
                                    binSize='1h',
                                    symbol=symbol,
                                    startTime=starttime,
                                    filter=f,
                                    count=1000,
                                    reverse=False,
                                    partial=includepartial).response().result

        # convert bitmex dict to df
        df = pd.io.json.json_normalize(result)
        usecols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close']
        newcols = ['Symbol', 'CloseTime', 'Open', 'High', 'Low', 'Close']
        df = df[usecols]
        df.columns = newcols
        df.CloseTime = df.CloseTime.astype('datetime64[ns]')

        if includepartial:
            self.partialcandle = df.tail(1).copy().reset_index(drop=True) # save last as df

            if not retainpartial:
                df.drop(df.index[-1], inplace=True)

        return df
    
    def printit(self, result):
        print(json.dumps(result, default=str, indent=4))

def comparestate(strat, pos):
    # return TRUE if side is GOOD
    # Could also check current contracts?
    # only works for trend, don't use for now
    contracts = pos['currentQty']
    side = f.side(contracts)

    ans = True if side == 0 or strat.getSide() == side else False

    if not ans:
        err = '{}: {}, expected: {}'.format(strat.sym.symbolshort, side, strat.getSide())
        f.discord(err)

    return ans

def compareorders(theo=[], actual=[]):
    
    matched, missing, notmatched = [], [], []
    m1, m2 = {}, {}

    for i, order in enumerate(theo):
        m1[order.key] = i
    
    for i, order in enumerate(actual):
        m2[order['key']] = i

    for order in theo:
        if order.key in m2:
            order.matched = True # do we need this if using a 'matched' list?
            order.intakelivedata(actual[m2[order.key]])
            matched.append(order)
        else:
            missing.append(order)

    for order in actual:
        if not order['key'] in m1:
            notmatched.append(order)

    # notmatched = [order for order in actual if not order['key'] in m1]
    
    return matched, missing, notmatched

def checkmatched(matched):
    
    amend = []

    for order in matched:
        ld = order.livedata
        checkprice = ld['price'] if ld['ordType'] == 'Limit' else ld['stopPx']
        side = 1 if ld['side'] == 'Buy' else -1

        if not (order.price == checkprice and
                abs(order.contracts) == ld['orderQty'] and
                order.side == side):
            amend.append(order)
        
    return amend

def refresh_gsheet_balance():
    sht = f.getGoogleSheet()
    ws = sht.worksheet_by_title('Bitmex')
    df = ws.get_as_df(start='A1', end='J14')
    lst = list(df['Sym'].dropna())
    syms = []

    df2 = pd.read_csv(os.path.join(f.currentdir(), 'symbols.csv'))
    for row in df2.itertuples():
        if row.symbolshort in lst:
            syms.append(c.Backtest(symbol=row.symbol))

    u = User()
    writeUserBalanceGoogle(syms, u, sht=sht, ws=ws, df=df)
    
def checkfilledorders(minutes=5, refresh=True, u=None):
    
    if u is None: u = User()
    starttime = date.utcnow() + delta(minutes=minutes * -1)
    orders = u.getFilledOrders(starttime=starttime)

    if orders:
        df = pd.read_csv(os.path.join(f.currentdir(), 'symbols.csv'))
        
        lst, syms, templist = [], [], []
        nonmarket, nonpersonal = False, False

        for o in orders:
            symbol = o['symbol']

            # check for bot orders
            if len(o['clOrdID']) > 0: nonpersonal = True

            # check for non-market buys
            if not o['ordType'] == 'Market':
                nonmarket = True

                # need to have all correct symbols in symbols.csv
                if not symbol in templist:
                    templist.append(symbol)
                    syms.append(c.Backtest(symbol=symbol))             

            modi = 1 if o['side'] == 'Buy' else -1
            vals = df[df['symbolbitmex']==symbol]['symbolshort'].values
            
            # TODO: Make this a property of order
            if len(vals) > 0: symshort = vals[0]
            lst.append('{} | {} {:,} at ${:,} | {}'.format(
                    symshort,
                    o['side'],
                    modi * int(o['orderQty']),
                    o['price'],
                    '-'.join(o['clOrdID'].split('-')[1:3])))
            
        # write balance to google sheet, EXCEPT on market buys
        if nonmarket and nonpersonal and refresh:
            TopLoop(u=u, partial=True)
            # writeUserBalanceGoogle(syms, u, preservedf=True)

        msg = '\n'.join(lst)
        f.discord(msg=msg+'\n@here', channel='orders')

def writeUserBalanceGoogle(syms, u, sht=None, ws=None, preservedf=False, df=None):
    
    if sht is None:
        sht = f.getGoogleSheet()
    if ws is None:
        ws = sht.worksheet_by_title('Bitmex')

    if df is None:
        if not preservedf:
            df = pd.DataFrame(columns=['Sym','Size','Entry','Last',	'Pnl', '%',	'ROE','Value', 'Dur', 'Conf'], index=range(13))
        else:
            df = ws.get_as_df(start='A1', end='J14')

    i = 0 # should pull gRow from google sheet first
    u.setOrders()
    u.setPositions()

    for sym in syms:
        symbol = sym.symbolbitmex
        figs = sym.decimalfigs
        pos = u.getPosition(symbol)
        
        df.at[i, 'Sym'] = sym.symbolshort
        if sym.tradingenabled:
            df.at[i, 'Size'] = pos['currentQty']
            df.at[i, 'Entry'] = round(pos['avgEntryPrice'], figs)
            df.at[i, 'Last'] = round(pos['lastPrice'], figs)
            df.at[i, 'Pnl'] = round(pos['unrealisedPnl'] / u.div, 3)
            df.at[i, '%'] = f.percent(pos['unrealisedPnlPcnt'])
            df.at[i, 'ROE'] = f.percent(pos['unrealisedRoePcnt'])
            df.at[i, 'Value'] = pos['maintMargin'] / u.div

        if not sym.startdate is None:
            strat = sym.strats[0]
            t = strat.trades[-1]
            df.at[i, 'Dur'] = t.duration()
            df.Conf[i] = t.conf

        i += 1
    
    # set profit/balance
    df.Size[9] = u.unrealizedpnl
    df.Entry[9] = u.totalbalancemargin

    # set current time
    df.Sym[12] = 'Last:'
    df.Size[12] = date.strftime(date.utcnow(), f.TimeFormat(mins=True))

    df = pd.concat([df, u.dfOrders(refresh=True)], axis=1)

    ws.set_dataframe(df, (1,1), nan='')
    return df
    
def TopLoop(u=None, partial=False):
    # run every 1 hour, or when called by checkfilledorders()

    # Google - get user/position info
    sht = f.getGoogleSheet()
    g_usersettings = sht.worksheet_by_title('UserSettings').get_all_records()
    dfsym = pd.read_csv(os.path.join(f.currentdir(), 'symbols.csv'))
    g_user = g_usersettings[0] #0 is jayme
    syms = []

    # Bitmex - get user/position info
    if u is None: u = User()
    u.setPositions()
    u.setOrders()
    u.reservedbalance = g_user['Reserved Balance'] # could just pass g_user to User()
    
    # TODO: filter dfall to only symbols needed, don't pull everything from db
    # use 'WHERE symbol in []', try pypika
    # Only using XBTUSD currently
    startdate, daterange = date.now().date() + delta(days=-15), 30
    dfall = f.getDataFrame(symbol='XBTUSD', startdate=f.startvalue(startdate), enddate=f.enddate(startdate, daterange))

    for row in dfsym.itertuples():
        if not row.symbol=='XBTUSD': continue
        try:
            # match google user with bitmex position, add %balance
            weight = float(g_user[row.symbolshort].strip('%')) / 100
            pos = u.getPosition(row.symbolbitmex)
            pos['percentbalance'] = weight

            symbol = row.symbol
            df = dfall[dfall.Symbol==symbol].reset_index(drop=True)
            
            # TREND_REV
            speed = (16, 6)
            norm = (0.004, 0.024)
            strat = c.Strat_TrendRev(speed=speed, norm=norm)
            strat.stoppercent = -0.03
            strats = [strat]

            sym = c.Backtest(symbol=symbol, startdate=startdate, strats=strats, row=row, df=df, partial=partial, u=u)
            if weight <= 0:
                sym.tradingenabled = False #this should come from strat somehow
            sym.decidefull()
            syms.append(sym)

            if sym.tradingenabled:
                actual = u.getOrders(sym.symbolbitmex)
                theo = strat.finalOrders(u, weight)
                
                matched, missing, notmatched = compareorders(theo, actual)

                u.cancelbulk(notmatched)
                u.amendbulk(checkmatched(matched))
                u.placebulk(missing)

        except:
            f.senderror(symbol)

    writeUserBalanceGoogle(syms, u, sht)
