import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta as delta
from datetime import timezone as tz
from pathlib import Path

import numpy as np
import pandas as pd
from bitmex import bitmex

from . import (
    functions as f,
    backtest as bt)
from .database import db
from .strategies import (trendrev, sfp)

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

        df = pd.read_csv(f.topfolder / 'data/ApiKeys/bitmex.csv')
        user = 'jayme' if not test else 'testnet'
        api_key = df.apikey.loc[df.user == user].values[0]
        api_secret = df.apisecret.loc[df.user == user].values[0]

        self.client = bitmex(test=test, api_key=api_key, api_secret=api_secret)
        self.set_total_balance()

    def balance(self):
        return self.totalbalancewallet - self.reservedbalance

    def check_request(self, request, retries=0):
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
                return self.check_request(request=request, retries=retries)
            else:
                f.send_error('{}: {}\n{}'.format(status, response.result, request.future.request.data))
        except:
            # request.prepare() #TODO: this doesn't work
            f.send_error('HTTP Error: {}'.format(request.future.request.data))

    def get_position(self, symbol, refresh=False):
        if self.positions is None or refresh:
            self.set_positions()
        
        return list(filter(lambda x: x['symbol']==symbol, self.positions))[0]

    def set_positions(self, fltr=''):
        self.positions = self.client.Position.Position_get(filter=fltr).result()[0]

    def open_contracts(self):
        self.set_positions()
        m = {}
        for p in self.positions:
            m[p['symbol']] = p['currentQty']

        return m

    def df_orders(self, symbol=None, newonly=True, refresh=False):
        orders = self.get_orders(symbol=symbol, newonly=newonly, refresh=refresh)
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

    def get_order_by_key(self, key):
        if self.orders is None:
            self.set_orders(newonly=True)
        
        # Manual orders don't have a key, not unique
        orders = list(filter(lambda x: 'key' in x.keys(), self.orders))
        orders = list(filter(lambda x: x['key']==key, orders))

        if orders:
            return orders[0] # assuming only one order will match given key
        else:
            return defaultdict(type(None))

    def get_filled_orders(self, symbol='', starttime=None):

        if starttime is None:
            starttime = dt.utcnow() + delta(days=-7)

        fltr = dict(ordStatus='Filled')

        self.set_orders(fltr=fltr, newonly=False, starttime=starttime, reverse=False)
        return self.orders
        
    def get_orders(self, symbol=None, newonly=True, botonly=False, manualonly=False, refresh=False, count=100):
        if self.orders is None or refresh:
            self.orders = []
            self.set_orders(newonly=newonly, count=count)
        
        orders = self.orders

        if not symbol is None:
            orders = list(filter(lambda x: x['symbol']==symbol, orders))

        if botonly:
            orders = list(filter(lambda x: x['manual']==False, orders))

        if manualonly:
            orders = list(filter(lambda x: x['manual']==True, orders))

        return orders
            
    def add_order_info(self, orders):
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
            
    def set_orders(self, fltr={}, newonly=True, count=100, starttime=None, reverse=True):
        if newonly:
            fltr['ordStatus'] = 'New'

        fltr = json.dumps(fltr)

        ep = self.client.Order
        orders = ep.Order_getOrders(
                                filter=fltr,
                                reverse=reverse,
                                count=count,
                                startTime=starttime).response().result
        
        self.orders = self.add_order_info(orders)

    def funding_rate(self, symbol='XBTUSD'):
        result = self.client.Instrument.Instrument_get(symbol='XBTUSD').response().result[0]

        rate = result['fundingRate']
        hrs = int((result['fundingTimestamp'] - f.timenow().replace(tzinfo=tz.utc)).total_seconds() / 3600)

        return rate, hrs

    def set_total_balance(self):
        div = self.div
        res = self.check_request(self.client.User.User_getMargin(currency='XBt'))
        self.availablemargin = res['excessMargin'] / div # total available/unused > only used in postOrder
        self.totalbalancemargin = res['marginBalance'] / div # unrealized + realized > don't actually use 
        self.totalbalancewallet = res['walletBalance'] / div # realized
        self.unrealizedpnl = res['unrealisedPnl'] / div
        self.prevpnl = res['prevRealisedPnl'] / div

    def amend_bulk(self, amendorders):
        # accept list of Jambot.Order() objects, convert and send amend command to bitmex.
        try:
            orders = [order.amend_order() for order in amendorders]
            if not orders: return
            return self.check_request(self.client.Order.Order_amendBulk(orders=json.dumps(orders)))
        except:
            msg = ''
            for order in amendorders:
                msg += json.dumps(order.amend_order()) + '\n'
            f.send_error(msg)

    def place_manual(self, contracts, price=None, ordtype='Limit', symbol='XBTUSD'):

        if price is None:
            ordtype='Market'
        
        order = bt.Order(price=price,
                        contracts=contracts,
                        ordtype=ordtype,
                        symbol=symbol)
        
        print('Sending new_order:')
        display(order.new_order())
        
        o = self.place_bulk(placeorders=order)[0]
        display(f.useful_keys(o))
    
    def place_bulk(self, placeorders):
        orders = []
        if not isinstance(placeorders, list): placeorders = [placeorders]
        l = len(placeorders)

        for order in placeorders:
            if l > 1 and order.new_order().get('ordType', '') == 'Market':
                self.place_bulk(order)
            else:
                orders.append(order.new_order())
                    
        if not orders: return
        try:
            result = self.add_order_info(
                        self.check_request(
                            self.client.Order.Order_newBulk(
                                orders=json.dumps(orders))))
            
            for o in result:
                if o['ordStatus'] == 'Canceled':
                    f.send_error(msg='***** ERROR: Order CANCELLED! \n{} \n{}'.format(f.useful_keys(o), o['text']))

            return result
        except:
            msg = 'Cant place orders: \n'
            for order in placeorders:
                msg += str(order.new_order()).replace(', ', '\n') + '\n\n'
            f.send_error(msg=msg)
    
    def close_position(self, symbol='XBTUSD'):
        try:
            # m = dict(symbol='XBTUSD', execInst='Close', ordType='Market')
            self.check_request(
                self.client.Order.Order_new(symbol='XBTUSD', execInst='Close'))
            # self.check_request(
            #     self.client.Order.Order_newBulk(
            #         orders=json.dumps(m)))
        except:
            f.send_error(msg='ERROR: Could not close position!')
    
    def cancel_manual(self):
        orders = self.get_orders(refresh=True, manualonly=True)
        self.cancel_bulk(orders=orders)
    
    def cancel_bulk(self, orders):
        # only need ordID to cancel
        if not isinstance(orders, list): orders = [orders]
        orders = [order['orderID'] for order in orders]
        if not orders: return
        return self.check_request(self.client.Order.Order_cancel(orderID=json.dumps(orders)))
        
    def get_partial(self, symbol):
        timediff = 0
        if not self.partialcandle is None:
            timediff = (self.partialcandle.Timestamp[0] - f.timenow()).seconds

        if (timediff > 7200
            or self.partialcandle is None
            or not self.partialcandle.Symbol[0] == symbol):
                self.set_partial(symbol=symbol)
        
        return self.partialcandle
        
    def set_partial(self, symbol):
        # call only partial candle from bitmex, save to self.partialcandle
        starttime = dt.utcnow() + delta(hours=-2)
        self.get_candles(symbol=symbol, starttime=starttime)
    
    def append_partial(self, df):
        # partial not built to work with multiple symbols, need to add partials to dict
        # Append partialcandle df to df from SQL db

        symbol = df.Symbol[0]
        dfpartial = self.get_partial(symbol=symbol)

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

    def get_candles(self, symbol='', starttime=None, fltr='', retainpartial=False, includepartial=True, interval=1, count=1000, pages=100):

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
            
            resultcount = len(result)
            print(f'page: {start // 1000}, start: {start}, ts: {starttime + delta(hours=start)} results: {resultcount}')
            lst.extend(result)
            start += 1000

        # convert bitmex dict to df
        df = pd.json_normalize(lst)
        df.columns = [x.capitalize() for x in df.columns]
        df.Timestamp = df.Timestamp.astype('datetime64[ns]') + offset * -1

        if interval == 15:
            df = self.resample(df=df, includepartial=includepartial)
        

        # keep all volume in terms of BTC
        cols = ['Interval', 'Symbol', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'VolBTC']

        df = df \
            .assign(
                Interval=interval,
                VolBTC=lambda x: np.where(x.Symbol=='XBTUSD', x.Homenotional, x.Foreignnotional)) \
            [cols]
  
        if includepartial:
            self.partialcandle = df.tail(1).copy().reset_index(drop=True) # save last as df

            if not retainpartial:
                df.drop(df.index[-1], inplace=True)

        return df
    
    def printit(self, result):
        print(json.dumps(result, default=str, indent=4))

def compare_state(strat, pos):
    # return TRUE if side is GOOD
    # Could also check current contracts?
    # only works for trend, don't use for now
    contracts = pos['currentQty']
    side = f.side(contracts)

    ans = True if side == 0 or strat.get_side() == side else False

    if not ans:
        err = '{}: {}, expected: {}'.format(strat.sym.symbolshort, side, strat.get_side())
        f.discord(err)

    return ans

def compare_orders(theo=[], actual=[], show=False):
    
    matched, missing, notmatched = [], [], []
    m1, m2 = {}, {}
    # TODO: this could be done with filtering

    for i, o in enumerate(theo):
        m1[o.key] = i
    
    for i, o in enumerate(actual):
        m2[o['key']] = i

    for o in theo:
        if o.key in m2:
            o.intake_live_data(actual[m2[o.key]])
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

def check_matched(matched, show=False):
    
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
                display(order.amend_order())
        
    return amend

def refresh_gsheet_balance(u=None):
    sht = f.get_google_sheet()
    ws = sht.worksheet_by_title('Bitmex')
    df = ws.get_as_df(start='A1', end='J15')
    lst = list(df['Sym'].dropna())
    syms = []

    df2 = pd.read_csv(os.path.join(f.topfolder, 'data/symbols.csv'))
    for row in df2.itertuples():
        if row.symbolshort in lst:
            syms.append(bt.Backtest(symbol=row.symbol, row=row))

    if u is None: u = User()
    write_balance_google(syms, u, sht=sht, ws=ws, df=df)
    
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
    
    if msg: f.discord(msg=msg, channel='sfp')

def check_filled_orders(minutes=5, refresh=True, u=None):
    if u is None: u = User()
    starttime = dt.utcnow() + delta(minutes=minutes * -1)
    orders = u.get_filled_orders(starttime=starttime)

    if orders:
        df = pd.read_csv(Path(f.topfolder) / 'data/symbols.csv')
        
        lst, syms, templist = [], [], []
        nonmarket = False

        for o in orders:
            symbol, name = o['symbol'], o['name']
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
                    syms.append(bt.Backtest(symbol=symbol))       

            ordprice = f' ({price})' if not price == avgpx else ''
            stats = f' | Bal: {u.totalbalancemargin:.3f} | PnL: {u.prevpnl:.3f}' if any(s in name for s in ('close', 'stop')) else ''

            lst.append('{} | {} {:,} at ${:,}{} | {}{}'.format(
                    symshort,
                    o['sideStr'],
                    o['contracts'],
                    avgpx,
                    ordprice,
                    name,
                    stats))
            
        # write balance to google sheet, EXCEPT on market buys
        if nonmarket and refresh:
            run_toploop(u=u, partial=True)
            # write_balance_google(syms, u, preservedf=True)

        msg = '\n'.join(lst)
        f.discord(msg=msg+'\n@here', channel='orders')
        # return msg

def write_balance_google(syms, u, sht=None, ws=None, preservedf=False, df=None):
    
    if sht is None:
        sht = f.get_google_sheet()
    if ws is None:
        ws = sht.worksheet_by_title('Bitmex')

    if df is None:
        if not preservedf:
            df = pd.DataFrame(columns=['Sym','Size','Entry','Last',	'Pnl', '%',	'ROE','Value', 'Dur', 'Conf'], index=range(14))
        else:
            df = ws.get_as_df(start='A1', end='J15')

    u.set_positions()

    for i, sym in enumerate(syms):
        symbol = sym.symbolbitmex
        figs = sym.decimalfigs
        pos = u.get_position(symbol)
        
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
    u.set_total_balance()
    df.at[9, 'Size'] = u.unrealizedpnl
    df.at[9, 'Entry'] = u.totalbalancemargin

    # set funding rate
    rate, hrs = u.funding_rate()
    df.at[12, 'Sym'] = 'Funding:'
    df.at[12, 'Size'] = f.percent(rate)
    df.at[12, 'Entry'] = hrs
    
    # set current time
    df.at[13, 'Sym'] = 'Last:'
    df.at[13, 'Size'] = dt.strftime(dt.utcnow(), f.time_format(mins=True))

    # concat last 10 trades for google sheet
    sym = list(filter(lambda x: x.symbol=='XBTUSD', syms))[0]
    if sym.strats:
        dfTrades = sym.strats[0].result(last=10).drop(columns=['N', 'Contracts', 'Bal'])
        dfTrades.Timestamp = dfTrades.Timestamp.dt.strftime('%Y-%m-%d %H')
        dfTrades.Pnl = dfTrades.Pnl.apply(lambda x: f.percent(x))
        dfTrades.PnlAcct = dfTrades.PnlAcct.apply(lambda x: f.percent(x))
    else:
        dfTrades = ws.get_as_df(start='Q1', end='Y14') # df.loc[:, 'Timestamp':'PnlAcct']

    df = pd.concat([df, u.df_orders(refresh=True), dfTrades], axis=1)
    ws.set_dataframe(df, (1,1), nan='')
    # return df
    
def run_toploop(u=None, partial=False, dfall=None):
    # run every 1 hour, or when called by check_filled_orders()

    # Google - get user/position info
    sht = f.get_google_sheet()
    g_usersettings = sht.worksheet_by_title('UserSettings').get_all_records()
    dfsym = pd.read_csv(Path(f.topfolder) / 'data/symbols.csv')
    g_user = g_usersettings[0] #0 is jayme
    syms = []

    # Bitmex - get user/position info
    if u is None: u = User()
    u.set_positions()
    u.set_orders()
    u.reservedbalance = g_user['Reserved Balance'] # could just pass g_user to User()
    
    # TODO: filter dfall to only symbols needed, don't pull everything from db
    # use 'WHERE symbol in []', try pypika
    # Only using XBTUSD currently
    startdate = f.timenow() + delta(days=-15)
    if dfall is None:
        dfall = db.get_dataframe(symbol='XBTUSD', startdate=startdate, interval=1)
        
    for row in dfsym.itertuples():
        if not row.symbol=='XBTUSD': continue
        try:
            # match google user with bitmex position, add %balance
            weight = float(g_user[row.symbolshort].strip('%')) / 100
            pos = u.get_position(row.symbolbitmex)
            pos['percentbalance'] = weight

            symbol = row.symbol
            # NOTE may need to set Timestamp/Symbol index when using more than just XBTUSD
            df = dfall[dfall.Symbol==symbol] #.reset_index(drop=True)
            
            # TREND_REV
            speed = (16, 6)
            norm = (0.004, 0.024)
            strat = trendrev.Strategy(speed=speed, norm=norm)
            strat.stoppercent = -0.03
            strats = [strat]

            sym = bt.Backtest(symbol=symbol, startdate=startdate, strats=strats, row=row, df=df, partial=partial, u=u)
            if weight <= 0:
                sym.tradingenabled = False #this should come from strat somehow
            sym.decide_full()
            syms.append(sym)

            if sym.tradingenabled:
                actual = u.get_orders(sym.symbolbitmex, botonly=True)
                theo = strat.final_orders(u, weight)
                
                matched, missing, notmatched = compare_orders(theo, actual, show=True)

                u.cancel_bulk(notmatched)
                u.amend_bulk(check_matched(matched, show=True))
                u.place_bulk(missing)

        except:
            f.send_error(symbol)

    write_balance_google(syms, u, sht)
