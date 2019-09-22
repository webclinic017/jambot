import json
import os
import time
from collections import defaultdict
from datetime import datetime as date
from datetime import timedelta as delta

import pandas as pd
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
        self.unrealizedpnl = 0
        self.orders = None
        self.positions = None
        self.div = 100000000
        
        if not test:
            api_key = ''
            api_secret = ''
        else:
            api_key = 'ug95yWa1yCZzt7P-pXubAyEc'
            api_secret = 'Ry3Terz_5M9ldNKP8tNamv84dUhxkp-P5Re9_ljKw8raIwgX'

        self.client = bitmex(test=test, api_key=api_key, api_secret=api_secret)
        self.setTotalBalance()

    def checkrequest(self, request, retries=0):
        # request = bravado.http_future.HttpFuture
        # response = bravado.response.BravadoResponse
        response = request.response()
        status = response.metadata.status_code
        
        if status != 503 or retries >= 6:
            return response.result
        else:
            backoff = 0.5
            retries += 1
            sleeptime = backoff * (2 ** retries - 1)
            time.sleep(sleeptime)
            # print('retrying in: {}'.format(sleeptime))
            return self.checkrequest(request=request, retries=retries)

    def getPosition(self, symbol=None):
        if self.positions is None:
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

    def getOrderByKey(self, key):
        if self.orders is None:
            self.setOrders(newonly=True)
        
        if key in self.orderkeysdict:
            return self.orders[self.orderkeysdict[key]]       
        else:
            return defaultdict(type(None))

    def getFilledOrders(self, symbol='', starttime=None):

        if starttime is None:
            starttime = date.utcnow() + delta(days=-365)

        fltr = json.dumps(dict(ordStatus='Filled'))

        return self.client.Order.Order_getOrders(symbol=symbol,
                                                filter=fltr,
                                                count=100,
                                                startTime=starttime).response().result
        
    def getOrders(self, symbol, newonly=True, refresh=False):
        if self.orders is None or refresh:
            self.orders = []
            self.setOrders(newonly=newonly)
        
        return [self.orders[i] for i in self.orderdict[symbol]]
            
    def setOrders(self, symbol='', fltr={}, newonly=True):
        self.orderdict = defaultdict(list)
        self.orderkeysdict = {}
        
        if newonly:
            fltr['ordStatus'] = 'New'

        fltr = json.dumps(fltr)

        self.orders = self.client.Order.Order_getOrders(symbol=symbol, filter=fltr).response().result
        for i, order in enumerate(self.orders):
            
            order['side'] = 1 if order['side'] == 'Buy' else -1

            # add key to the order
            if not order['clOrdID'] == '':
                order['key'] = '-'.join(order['clOrdID'].split('-')[:-1])
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
            if len(orders) == 0: return
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
                    
        # orders = [order.neworder() for order in placeorders]
        if len(orders) == 0: return
        return self.checkrequest(self.client.Order.Order_newBulk(orders=json.dumps(orders)))
    
    def cancelbulk(self, cancelorders):
        # only need ordID to cancel
        orders = [order['orderID'] for order in cancelorders]
        if len(orders) == 0: return
        return self.checkrequest(self.client.Order.Order_cancel(orderID=json.dumps(orders)))
    
    def getCandles(self, symbol='', startTime=None, f='', reverse=True):
        return self.client.Trade.Trade_getBucketed(binSize='1h', symbol=symbol, startTime=startTime, filter= f,reverse=reverse).response().result
    
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

def checkfilledorders():

    u = User()
    starttime = date.utcnow() + delta(minutes=-5)
    orders = u.getFilledOrders(starttime=starttime)

    df = pd.read_csv(os.path.join(f.currentdir(), 'symbols.csv'))
    
    lst = []
    for o in orders:
        modi = 1 if o['side'] == 'Buy' else -1
        symshort = df[df['symbolbitmex']==o['symbol']]['symbolshort'].values[0]
        lst.append('{} {}: {:,}'.format(symshort, o['side'], modi * int(o['orderQty'])))

    msg = '\n'.join(lst)
    # print(msg)
    f.discord(msg=msg, channel='orders')


def writeUserBalanceGoogle(syms, u, sht):
    
    ws = sht.worksheet_by_title('Bitmex')

    df = pd.DataFrame(columns=['Sym','Size','Entry','Last',	'Pnl', '%',	'ROE','Value', 'Enter', 'Exit', 'Stop', 'Enter_Next', 'Stop_Next', 'Dur'], index=range(13))

    i = 0
    for sym in syms:
        symbol = sym.symbolbitmex
        figs = sym.decimalfigs
        pos = u.getPosition(symbol)
        strat = sym.strats[0]
        t = strat.trades[-1]
        side = t.side
        side_next = side * -1

        u.setOrders()
        u.setPositions()        
        
        df['Sym'][i] = sym.symbolshort
        if sym.tradingenabled:
            df['Size'][i] = pos['currentQty']
            df['Entry'][i] = round(pos['avgEntryPrice'], figs)
            df['Last'][i] = round(pos['prevClosePrice'], figs)
            df['Pnl'][i] = round(pos['unrealisedPnl'] / u.div, 3)
            df['%'][i] = f.percent(pos['unrealisedPnlPcnt'])
            df['ROE'][i] = f.percent(pos['unrealisedRoePcnt'])
            df['Value'][i] = pos['maintMargin'] / u.div
            df['Dur'][i] = t.duration()

            # df['Stop Buy'][i] = u.getOrderByKey(sym.symbolbitmex, 'stopbuy')['stopPx']
            # df['Stop Close'][i] = u.getOrderByKey(sym.symbolbitmex, 'stopclose')['stopPx']
            if strat.name == 'trendrev':
                df['Enter'][i] = u.getOrderByKey(f.key(symbol, 'limitopen', side, 1))['price']
                df['Exit'][i] = u.getOrderByKey(f.key(symbol, 'limitclose', side * -1, 3))['price']
                df['Stop'][i] = u.getOrderByKey(f.key(symbol, 'stop', side * -1, 2))['stopPx']
                df['Enter_Next'][i] = u.getOrderByKey(f.key(symbol, 'limitopen', side_next, 1))['price']
                df['Stop_Next'][i] = u.getOrderByKey(f.key(symbol, 'stop', side_next * -1, 2))['stopPx']               
            elif strat.name == 'chop':
                orders = u.getOrders(symbol, refresh=True)
                ordcount, stopcount, tpcount = 0, 0, 0
                
                for order in orders:
                    name = order['clOrdID'].split('-')[1]
                    
                    if name[0] == 'o':
                        ordcount += 1
                    elif name[0] == 's':
                        stopcount += 1
                    elif name[0] == 't':
                        tpcount += 1

                strat = sym.strats[0]
                df['Filled'][i] = strat.getTrade(strat.tradecount()).allfilled() if strat.getSide() != 0 else ''

                df['Ord'][i] = ordcount
                df['Stop'][i] = stopcount
                df['Tp'][i] = tpcount

        i += 1
    
    # set profit/balance
    df['Size'][9] = u.unrealizedpnl
    df['Entry'][9] = u.totalbalancemargin

    # set current time
    df['Size'][12] = date.strftime(date.utcnow(), f.TimeFormat(mins=True))

    ws.set_dataframe(df, (1,1), nan='')
    return df
    
def TopLoop():
    # run every 1 hour

    # Google - get user/position info
    sht = f.getGoogleSheet()
    g_usersettings = sht.worksheet_by_title('UserSettings').get_all_records()
    dfsym = pd.read_csv(os.path.join(f.currentdir(), 'symbols.csv'))
    g_user = g_usersettings[0] #0 is jayme
    syms = []

    # Bitmex - get user/position info
    u = User()
    u.setPositions() # also write back to google sheet
    u.setOrders()
    
    startdate, daterange = date.now().date() + delta(days=-15), 30
    dfall = f.getDataFrame(startdate=f.startvalue(startdate), enddate=f.enddate(startdate, daterange))

    for row in dfsym.itertuples():
        if not row.symbol == 'XBTUSD': continue
        try:
            # match google user with bitmex position, add %balance
            weight = float(g_user[row.symbolshort].strip('%')) / 100
            pos = u.getPosition(row.symbolbitmex)
            pos['percentbalance'] = weight

            symbol = row.symbol
            df = dfall[dfall.Symbol==symbol].reset_index(drop=True)
            
            strats = []
            # trend = c.Strat_Trend(againstspeed=row.against, withspeed=row.withspeed)
            # strats.append(trend)
            
            # speed = (row.against2, row.with2)
            # norm = (row.lowernormal, row.uppernormal)
            # chop = c.Strat_Chop(speed=speed, norm=norm)
            # strats.append(chop)

            # TREND_REV
            speed = (16, 6)
            norm = (0.02, 0.08)
            trend = c.Strat_TrendRev(speed=speed, norm=norm)
            strats.append(trend)

            sym = c.Backtest(symbol=symbol, startdate=startdate, strats=strats, row=row, df=df)
            if weight <= 0: sym.tradingenabled = False #this should come from strat somehow
            sym.decidefull()
            syms.append(sym)
            strat = sym.strats[0]

            if sym.tradingenabled:
                actual = u.getOrders(sym.symbolbitmex)
                theo = strat.finalOrders(u, weight)
                
                # if comparestate(strat, pos):
                matched, missing, notmatched = compareorders(theo, actual)
                # for order in matched:
                #     print('matched:{}'.format(order.amendorder()))
                # for order in missing:
                #     print('missing:{}'.format(order.neworder()))
                # for order in notmatched:
                #     print('notmatched:{}'.format(order))

                u.cancelbulk(notmatched)
                u.amendbulk(checkmatched(matched))
                u.placebulk(missing)

        except:
            f.senderror(symbol)

    writeUserBalanceGoogle(syms, u, sht)
    # return dfgoogle
