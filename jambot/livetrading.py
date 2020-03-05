import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime as date
from datetime import timedelta as delta
from datetime import timezone as tz
from pathlib import Path

import numpy as np
import pandas as pd
from bitmex import bitmex

import functions as f
import jambotclasses as c

try:
    from IPython.display import display
except ModuleNotFoundError:
    pass


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
        self.prevpnl = 0
        self.orders = None
        self.positions = None
        self.partialcandle = None
        self.div = 100000000

        df = pd.read_csv(Path(f.topfolder) / 'data/ApiKeys/bitmex.csv')
        user = 'jayme' if not test else 'testnet'
        api_key = df.apikey.loc[df.user == user].values[0]
        api_secret = df.apisecret.loc[df.user == user].values[0]

        self.client = bitmex(test=test, api_key=api_key, api_secret=api_secret)
        self.setTotalBalance()

    def balance(self):
        return self.totalbalancewallet - self.reservedbalance

    def checkrequest(self, request, retries=0):
        # type(request) = bravado.http_future.HttpFuture
        # type(response) = bravado.response.BravadoResponse
        # response = request.response(fallback_result=[400], exceptions_to_catch=bravado.exception.HTTPBadRequest)

        try:
            response = request.response(fallback_result='')
            status = response.metadata.status_code
            backoff = 0.5

            if status < 300:
                return response.result
            elif status == 503 and retries < 7:
                retries += 1
                sleeptime = backoff * (2 ** retries - 1)
                time.sleep(sleeptime)
                return self.checkrequest(request=request, retries=retries)
            else:
                f.senderror('{}: {}\n{}'.format(status, response.result, request.future.request.data))
        except:
            # request.prepare() #TODO: this doesn't work
            f.senderror('HTTP Error: {}'.format(request.future.request.data))

    def getPosition(self, symbol, refresh=False):
        if self.positions is None or refresh:
            self.setPositions()
        
        return list(filter(lambda x: x['symbol']==symbol, self.positions))[0]

    def setPositions(self, fltr=''):
        self.positions = self.client.Position.Position_get(filter=fltr).result()[0]

    def opencontracts(self):
        self.setPositions()
        m = {}
        for p in self.positions:
            m[p['symbol']] = p['currentQty']

        return m

    def dfOrders(self, symbol=None, newonly=True, refresh=False):
        orders = self.getOrders(symbol=symbol, newonly=newonly, refresh=refresh)
        cols = ['ordType', 'name', 'size', 'price', 'execInst', 'symbol']
        
        if not orders:
            df = pd.DataFrame(columns=cols, index=range(1))
        else:
            df = pd.json_normalize(orders)
            df['size'] = df.orderQty * df.side
            df['price'] = np.where(df.price > 0, df.price, df.stopPx)
        
        df = df.reindex(columns=cols).sort_values(
            by=['symbol', 'ordType', 'name'], 
            ascending=[False, True, True]).reset_index(drop=True)
        return df

    def getOrderByKey(self, key):
        if self.orders is None:
            self.setOrders(newonly=True)
        
        # Manual orders don't have a key, not unique
        orders = list(filter(lambda x: 'key' in x.keys(), self.orders))
        orders = list(filter(lambda x: x['key']==key, orders))

        if orders:
            return orders[0] # assuming only one order will match given key
        else:
            return defaultdict(type(None))

    def getFilledOrders(self, symbol='', starttime=None):

        if starttime is None:
            starttime = date.utcnow() + delta(days=-7)

        fltr = dict(ordStatus='Filled')

        self.setOrders(fltr=fltr, newonly=False, starttime=starttime, reverse=False)
        return self.orders
        
    def getOrders(self, symbol=None, newonly=True, botonly=False, manualonly=False, refresh=False, count=100):
        if self.orders is None or refresh:
            self.orders = []
            self.setOrders(newonly=newonly, count=count)
        
        orders = self.orders

        if not symbol is None:
            orders = list(filter(lambda x: x['symbol']==symbol, orders))

        if botonly:
            orders = list(filter(lambda x: x['manual']==False, orders))

        if manualonly:
            orders = list(filter(lambda x: x['manual']==True, orders))

        return orders
            
    def addOrderInfo(self, orders):
        if not isinstance(orders, list): orders = [orders]

        for o in orders:
            o['sideStr'] = o['side']
            o['side'] = 1 if o['side'] == 'Buy' else -1
            o['contracts'] = int(o['side'] * o['orderQty'])

            # add key to the order, excluding manual orders
            if not o['clOrdID'] == '':
                o['name'] = '-'.join(o['clOrdID'].split('-')[1:-1])
                
                if not 'manual' in o['clOrdID']:
                    o['key'] = '-'.join(o['clOrdID'].split('-')[:-1])
                    o['manual'] = False
                else:
                    o['manual'] = True
            else:
                o['name'] = '(manual)'
                o['manual'] = True

        return orders
            
    def setOrders(self, fltr={}, newonly=True, count=100, starttime=None, reverse=True):
        if newonly:
            fltr['ordStatus'] = 'New'

        fltr = json.dumps(fltr)

        ep = self.client.Order
        orders = ep.Order_getOrders(
                                filter=fltr,
                                reverse=reverse,
                                count=count,
                                startTime=starttime).response().result
        
        self.orders = self.addOrderInfo(orders)

    def fundingRate(self, symbol='XBTUSD'):
        result = self.client.Instrument.Instrument_get(symbol='XBTUSD').response().result[0]

        rate = result['fundingRate']
        hrs = int((result['fundingTimestamp'] - f.timenow().replace(tzinfo=tz.utc)).total_seconds() / 3600)

        return rate, hrs

    def setTotalBalance(self):
        div = self.div
        res = self.client.User.User_getMargin(currency='XBt').response().result
        self.availablemargin = res['excessMargin'] / div # total available/unused > only used in postOrder
        self.totalbalancemargin = res['marginBalance'] / div # unrealized + realized > don't actually use 
        self.totalbalancewallet = res['walletBalance'] / div # realized
        self.unrealizedpnl = res['unrealisedPnl'] / div
        self.prevpnl = res['prevRealisedPnl'] / div
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

    def placemanual(self, contracts, price=None, ordtype='Limit', symbol='XBTUSD'):

        if price is None:
            ordtype='Market'
        
        order = c.Order(price=price,
                        contracts=contracts,
                        ordtype=ordtype,
                        symbol=symbol)
        
        print('Sending neworder:')
        display(order.neworder())
        
        o = self.placebulk(placeorders=order)[0]
        display(f.usefulkeys(o))
    
    def placebulk(self, placeorders):
        orders = []
        if not isinstance(placeorders, list): placeorders = [placeorders]
        l = len(placeorders)

        for order in placeorders:
            if l > 1 and order.neworder().get('ordType', '') == 'Market':
                self.placebulk(order)
            else:
                orders.append(order.neworder())
                    
        if not orders: return
        try:
            result = self.addOrderInfo(
                        self.checkrequest(
                            self.client.Order.Order_newBulk(
                                orders=json.dumps(orders))))
            
            for o in result:
                if o['ordStatus'] == 'Canceled':
                    f.senderror(msg='***** ERROR: Order CANCELLED! \n{} \n{}'.format(f.usefulkeys(o), o['text']))

            return result
        except:
            msg = 'Cant place orders: \n'
            for order in placeorders:
                msg += str(order.neworder()).replace(', ', '\n') + '\n\n'
            f.senderror(msg=msg)
    
    def cancelmanual(self):
        orders = self.getOrders(refresh=True, manualonly=True)
        self.cancelbulk(orders=orders)
    
    def cancelbulk(self, orders):
        # only need ordID to cancel
        if not isinstance(orders, list): orders = [orders]
        orders = [order['orderID'] for order in orders]
        if not orders: return
        return self.checkrequest(self.client.Order.Order_cancel(orderID=json.dumps(orders)))
        
    def getpartial(self, symbol):
        timediff = 0
        if not self.partialcandle is None:
            timediff = (self.partialcandle.Timestamp[0] - f.timenow()).seconds

        if (timediff > 7200
            or self.partialcandle is None
            or not self.partialcandle.Symbol[0] == symbol):
                self.setpartial(symbol=symbol)
        
        return self.partialcandle
        
    def setpartial(self, symbol):
        # call only partial candle from bitmex, save to self.partialcandle
        starttime = date.utcnow() + delta(hours=-2)
        self.getCandles(symbol=symbol, starttime=starttime)
    
    def appendpartial(self, df):
        # partial not built to work with multiple symbols, need to add partials to dict
        # Append partialcandle df to df from SQL db

        symbol = df.Symbol[0]
        dfpartial = self.getpartial(symbol=symbol)

        return df.append(dfpartial, sort=False).reset_index(drop=True)

    def resample(self, df, includepartial=False):
        from collections import OrderedDict
        # convert 5min candles to 15min
        # need to only include groups of 3 > drop last 1 or 2 rows
        # remove incomplete candles, split into groups first

        gb = df.groupby('Symbol')
        lst = []

        for symbol in gb.groups:
            df = gb.get_group(symbol)
            
            if not includepartial:
                l = len(df)
                cut = l - (l  // 3) * 3
                if cut > 0: df = df[:cut * -1]

            lst.append(df.resample('15Min', on='Timestamp').agg(
                OrderedDict([
                    ('Symbol', 'first'),
                    ('Open', 'first'),
                    ('High', 'max'),
                    ('Low', 'min'),
                    ('Close', 'last')])))

        return pd.concat(lst).reset_index()

    def getCandles(self, symbol='', starttime=None, fltr='', retainpartial=False, includepartial=True, interval=1, count=1000, pages=100):

        if interval == 1:
            binsize = '1h'
            offset = delta(hours=1)
        elif interval == 15:
            binsize = '5m'
            offset = delta(minutes=5)

        if not starttime is None:
            starttime += offset

        resultcount = float('inf')
        start = 0
        lst = []

        while resultcount >= 1000 and start // 1000 <= pages:
            result = self.client.Trade.Trade_getBucketed(
                                        binSize=binsize,
                                        symbol=symbol,
                                        startTime=starttime,
                                        filter=fltr,
                                        count=count,
                                        start=start,
                                        reverse=False,
                                        partial=includepartial).response().result
            
            # print(start // 1000)
            resultcount = len(result)
            lst.extend(result)
            start += 1000

        # convert bitmex dict to df
        df = pd.json_normalize(lst)
        df.columns = [x.capitalize() for x in df.columns]
        df.Timestamp = df.Timestamp.astype('datetime64[ns]') + offset * -1

        if interval == 15:
            df = self.resample(df=df, includepartial=includepartial)
    
        df['Interval'] = interval
        df = df[['Interval', 'Symbol', 'Timestamp', 'Open', 'High', 'Low', 'Close']]

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

def compareorders(theo=[], actual=[], show=False):
    
    matched, missing, notmatched = [], [], []
    m1, m2 = {}, {}
    # TODO: this could be done with filtering

    for i, o in enumerate(theo):
        m1[o.key] = i
    
    for i, o in enumerate(actual):
        m2[o['key']] = i

    for o in theo:
        if o.key in m2:
            o.intakelivedata(actual[m2[o.key]])
            matched.append(o)
        else:
            missing.append(o)

    for o in actual:
        if not o['key'] in m1:
            notmatched.append(o)

    if show and not 'linux' in sys.platform:
        print('\nMatched: ')
        for o in matched: display(o.to_dict())
        print('\nMissing:')
        for o in missing: display(o.to_dict())
        print('\nNot matched:')
        for o in notmatched: display(o)
    
    return matched, missing, notmatched

def checkmatched(matched, show=False):
    
    amend = []

    for order in matched:
        ld = order.livedata
        checkprice = ld['price'] if ld['ordType'] == 'Limit' else ld['stopPx']

        if not (order.price == checkprice and
                order.contracts == ld['contracts'] and
                order.side == ld['side']):
            amend.append(order)
            
            if show and not 'linux' in sys.platform:
                print('\nAmmending:')
                print(order.name)
                print(order.price == checkprice, order.price, checkprice)
                print(order.contracts == ld['contracts'], order.contracts, ld['contracts'])
                print(order.side == ld['side'], order.side, ld['side'])
                display(order.amendorder())
        
    return amend

def refresh_gsheet_balance(u=None):
    sht = f.getGoogleSheet()
    ws = sht.worksheet_by_title('Bitmex')
    df = ws.get_as_df(start='A1', end='J15')
    lst = list(df['Sym'].dropna())
    syms = []

    df2 = pd.read_csv(os.path.join(f.topfolder, 'data/symbols.csv'))
    for row in df2.itertuples():
        if row.symbolshort in lst:
            syms.append(c.Backtest(symbol=row.symbol, row=row))

    if u is None: u = User()
    writeUserBalanceGoogle(syms, u, sht=sht, ws=ws, df=df)
    
def checksfp(df):
    # run every hour
    # get last 196 candles, price info only
    # create sfp object
    # check if current candle returns any sfp objects
    # send discord alert with the swing fails, and supporting info
    # 'Swing High to xxxx', 'swung highs at xxxx', 'tail = xx%'
    # if one candle swings highs and lows, go with... direction of candle? bigger tail?
        
    sfp = c.Strat_SFP()
    sfp.init(df=df)
    sfps = sfp.isSwingFail()
    msg = ''
    cdl = sfp.cdl
    stypes = dict(high=1, low=-1)

    for k in stypes.keys():
        lst = list(filter(lambda x: k in x['name'], sfps))
        side = stypes[k]
        
        if lst:
            msg += 'Swing {} to {} | tail = {:.0%}\n'.format(k.upper(),
                                                    cdl.getmax(side=side),
                                                    cdl.tailpct(side=side))
            for sfp in lst:
                msg += '    {} at: {}\n'.format(sfp['name'], sfp['price'])
    
    if msg: f.discord(msg=msg, channel='sfp')

def checkfilledorders(minutes=5, refresh=True, u=None):
    
    if u is None: u = User()
    starttime = date.utcnow() + delta(minutes=minutes * -1)
    orders = u.getFilledOrders(starttime=starttime)

    if orders:
        df = pd.read_csv(Path(f.topfolder) / 'data/symbols.csv')
        
        lst, syms, templist = [], [], []
        nonmarket = False

        for o in orders:
            symbol = o['symbol']
            row = df[df['symbolbitmex']==symbol]
            figs = 0
            if len(row) > 0:
                symshort = row['symbolshort'].values[0]
                figs = row['decimalfigs'].values[0]

            price = o['price']
            avgpx = round(o['avgPx'], figs)

            # check for non-market buys
            if not o['ordType'] == 'Market':
                nonmarket = True

                # need to have all correct symbols in symbols.csv
                if not symbol in templist:
                    templist.append(symbol)
                    syms.append(c.Backtest(symbol=symbol))       

            ordprice = f' ({price})' if not price == avgpx else ''

            lst.append('{} | {} {:,} at ${:,}{} | {}'.format(
                    symshort,
                    o['sideStr'],
                    o['contracts'],
                    avgpx,
                    ordprice,
                    o['name']))
            
        # write balance to google sheet, EXCEPT on market buys
        if nonmarket and refresh:
            TopLoop(u=u, partial=True)
            # writeUserBalanceGoogle(syms, u, preservedf=True)

        msg = '\n'.join(lst)
        f.discord(msg=msg+'\n@here', channel='orders')
        # return msg

def writeUserBalanceGoogle(syms, u, sht=None, ws=None, preservedf=False, df=None):
    
    if sht is None:
        sht = f.getGoogleSheet()
    if ws is None:
        ws = sht.worksheet_by_title('Bitmex')

    if df is None:
        if not preservedf:
            df = pd.DataFrame(columns=['Sym','Size','Entry','Last',	'Pnl', '%',	'ROE','Value', 'Dur', 'Conf'], index=range(14))
        else:
            df = ws.get_as_df(start='A1', end='J15')

    u.setPositions()

    for i, sym in enumerate(syms):
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

        if sym.strats:
            strat = sym.strats[0]
            t = strat.trades[-1]
            df.at[i, 'Dur'] = t.duration()
            df.Conf[i] = t.conf
    
    # set profit/balance
    u.setTotalBalance()
    df.at[9, 'Size'] = u.unrealizedpnl
    df.at[9, 'Entry'] = u.totalbalancemargin

    # set funding rate
    rate, hrs = u.fundingRate()
    df.at[12, 'Sym'] = 'Funding:'
    df.at[12, 'Size'] = f.percent(rate)
    df.at[12, 'Entry'] = hrs
    
    # set current time
    df.at[13, 'Sym'] = 'Last:'
    df.at[13, 'Size'] = date.strftime(date.utcnow(), f.TimeFormat(mins=True))

    # concat last 10 trades for google sheet
    sym = list(filter(lambda x: x.symbol=='XBTUSD', syms))[0]
    if sym.strats:
        dfTrades = sym.strats[0].result(last=10).drop(columns=['N', 'Contracts', 'Bal'])
        dfTrades.Timestamp = dfTrades.Timestamp.dt.strftime('%Y-%m-%d %H')
        dfTrades.Pnl = dfTrades.Pnl.apply(lambda x: f.percent(x))
        dfTrades.PnlAcct = dfTrades.PnlAcct.apply(lambda x: f.percent(x))
    else:
        dfTrades = ws.get_as_df(start='Q1', end='Y14') # df.loc[:, 'Timestamp':'PnlAcct']

    df = pd.concat([df, u.dfOrders(refresh=True), dfTrades], axis=1)
    ws.set_dataframe(df, (1,1), nan='')
    # return df
    
def TopLoop(u=None, partial=False, dfall=None):
    # run every 1 hour, or when called by checkfilledorders()

    # Google - get user/position info
    sht = f.getGoogleSheet()
    g_usersettings = sht.worksheet_by_title('UserSettings').get_all_records()
    dfsym = pd.read_csv(Path(f.topfolder) / 'data/symbols.csv')
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
    startdate = f.timenow() + delta(days=-15)
    if dfall is None:
        dfall = f.getDataFrame(symbol='XBTUSD', startdate=startdate, interval=1)
        
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
                actual = u.getOrders(sym.symbolbitmex, botonly=True)
                theo = strat.finalOrders(u, weight)
                
                matched, missing, notmatched = compareorders(theo, actual, show=True)

                u.cancelbulk(notmatched)
                u.amendbulk(checkmatched(matched, show=True))
                u.placebulk(missing)

        except:
            f.senderror(symbol)

    writeUserBalanceGoogle(syms, u, sht)
