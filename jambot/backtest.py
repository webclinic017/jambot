import os
from datetime import (datetime as dt, timedelta as delta)
from enum import Enum
from pathlib import Path
from time import time
from collections import defaultdict as dd

import numpy as np
import pandas as pd


from . import (
    functions as f,
    livetrading as live)
from .database import db

try:
    from IPython.display import display
except ModuleNotFoundError:
    pass


class Account():
    def __init__(self):
        self.balance = 1
        self.max = 0
        self.min = 1
        self.txns = list()
        self._df_balance = None

    def get_balance(self):
        if self.balance < 0.01:
            self.balance = 0.01
        # balance = self.balance if not self.balance < 0.01 else 0.01
        return self.balance

    def reset(self, balance=1):
        self.balance = balance

    def modify(self, xbt, timestamp):
        txn = Txn()
        txn.amount = xbt
        txn.timestamp = timestamp
        txn.balance_pre = self.balance
        txn.percentchange = round((xbt / self.balance), 3)

        self.txns.append(txn)
        self.balance = self.balance + xbt
        txn.balance_post = self.balance
        if self.balance > self.max: self.max = self.balance
        if self.balance < self.min: self.min = self.balance    

    def get_period_num(self, timestamp, period='month'):
        timestamp = f.check_date(timestamp)
        with f.Switch(period) as case:
            if case('month'):
                return timestamp.month
            elif case('week'):
                return timestamp.strftime("%V")

    def drawdown(self):
        drawdown = 0.
        max_seen = 1
        txmax_seen = self.txns[0]
        for txn in self.txns:
            val = txn.balance_pre
            if val > max_seen:
                max_seen = val
                txmax_seen = txn

            curdraw = 1 - val / max_seen
            if curdraw > drawdown:
                drawdown = curdraw
                txlow = txn
                txhigh = txmax_seen
        
        drawdates = '{:.2f} ({}) - {:.2f} ({})'.format(
            txhigh.balance_pre,
            dt.strftime(txhigh.timestamp, f.time_format()),
            txlow.balance_pre,
            dt.strftime(txlow.timestamp, f.time_format()))
        
        return drawdown * -1, drawdates
    
    def get_percent_change(self, balance, change):
        return f.percent(change / balance)

    def print_summary(self, period='month'):
        # TODO: make this include current month
        data = []
        cols = ['Period', 'AcctBalance', 'Change', 'PercentChange']

        periodnum = self.get_period_num(self.txns[0].timestamp, period)
        prevTxn = None
        prevBalance = 1
        change = 0.0

        for t in self.txns:     
            if self.get_period_num(t.timestamp, period) != periodnum:
                if prevTxn is None: prevTxn = t
                data.append([
                    periodnum,
                    '{:.{prec}f}'.format(t.balance_post, prec=3),
                    round(change, 3),
                    self.get_percent_change(prevBalance, change)
                ])
                prevTxn = t
                prevBalance = t.balance_post
                change = 0
                periodnum = self.get_period_num(t.timestamp, period)
            
            change += t.amount

        df = pd.DataFrame(data=data, columns=cols)
        display(df)

    @property
    def df_balance(self):
        if self._df_balance is None:
            m = dd(list)

            for t in self.txns:
                m['timestamp'].append(t.timestamp)
                m['balance'].append(t.balance_post)

            self._df_balance = pd.DataFrame.from_dict(m) \
                .set_index('timestamp')
        
        return self._df_balance

    def plot_balance(self, logy=False, title=None):
        """Show plot of account balance over time"""
        self.df_balance.plot(kind='line', y='balance', logy=logy, linewidth=0.75, color='cyan', title=title)

    def print_txns(self):
        data = []
        cols = ['Date', 'AcctBalance', 'Amount', 'PercentChange']
        for t in self.txns:
            data.append([
                '{:%Y-%m-%d %H}'.format(t.timestamp),
                '{:.{prec}f}'.format(t.balance_pre, prec=3),
                '{:.{prec}f}'.format(t.amount, prec=2),
                f.percent(t.percentchange)
            ])
        
        pd.options.display.max_rows = len(data)
        df = pd.DataFrame(data=data, columns=cols)
        display(df)
        pd.options.display.max_rows = 100

    def get_df(self):
        df = pd.DataFrame(columns=['Timestamp', 'Balance', 'PercentChange'])
        for i, t in enumerate(self.txns):
            df.loc[i] = [t.timestamp, t.balance_pre, t.percentchange]
        return df

class Backtest():
    def __init__(self, symbol, startdate=None, strats=[], daterange=365, df=None, row=None, account=None, partial=False, u=None, **kw):

        if not isinstance(strats, list): strats = [strats]

        if row is None:
            dfsym = pd.read_csv(Path(f.topfolder) / 'data/symbols.csv')
            dfsym = dfsym[dfsym['symbol']==symbol]
            row = list(dfsym.itertuples())[0]

        self.row = row
        self.symbolshort = row.symbolshort
        self.urlshort = row.urlshort
        self.symbolbitmex = row.symbolbitmex
        self.altstatus = bool(row.altstatus)
        self.decimalfigs = row.decimalfigs
        self.tradingenabled = True
        self.partial = partial

        self.symbol = symbol
        self.startdate = f.check_date(startdate)
        self.strats = strats

        # actual backtest, not just admin info
        if not startdate is None:
            if account == None:
                self.account = Account()

            self.i = 1
            self.candles = []
            
            if df is None:
                self.df = db.get_dataframe(symbol=symbol, startdate=startdate, daterange=daterange)
            else:
                self.df = df

            if partial:
                if u is None: u = live.User()
                self.df = u.append_partial(df)
           
            self.startrow = self.df.index.get_loc(startdate)

            for strat in self.strats:
                strat.init(sym=self)

    def init_candle(self, c):
        self.cdl = c
        self.candles.append(c)

    def decide_full(self):
        df = self.df
        length = len(df)
        for c in df.itertuples():
            self.init_candle(c=c)
            
            self.i = df.index.get_loc(c.Index)
            i = self.i
            if not i < self.startrow:
                if i == length: return

                for strat in self.strats:
                    strat.decide(c)
        
        if self.partial:
            self.strats[0].trades[-1].partial = True

    def print_final(self):
        style = self.result().style.hide_index()
        style.format({'Min': '{:.3f}',
                    'Max': '{:.3f}',
                    'Final': '{:.3f}',
                    'Drawdown': '{:.2%}'})
        display(style)

    def result(self):
        a = self.account
        strat = self.strats[0]

        drawdown, drawdates = a.drawdown()

        data = {
            'symbol': [self.symbol],
            'Min': [a.min],
            'Max': [a.max],
            'Final': [a.balance],
            'Drawdown': [drawdown],
            'Period': [drawdates],
            'Goodtrades': ['{}/{}/{}'.format(strat.good_trades(), strat.tradecount(), strat.unfilledtrades)]}

        return pd.DataFrame(data=data)

    def write_csv(self):
        self.df.to_csv('dfout.csv')
        self.account.get_df().to_csv('df2out.csv')

    def expected_orders(self):
        # don't use this yet.. maybe when we have combined strats?
        
        expected = []
        for strat in self.strats:
            for order in strat.final_orders():
                expected.append(order)
        
        return expected

class Confidence():
    # confidence uses list of signals with weights

    def __init__(self):
        self.signals = []
        self.trendsignals = []
        self.maxconf = 1.5
        self.minconf = 0.25
        self.startconf = 1
        
    def final(self, side, c):
        conf = self.startconf
        for signal in self.signals:
            conf *= signal.final(side=side, c=c)

        if conf > self.maxconf:
            conf = self.maxconf
        elif conf < self.minconf:
            conf = self.minconf

        return conf

    def add_signal(self, df, signals=None, trendsignals=None):
        self.signals.extend(signals or [])
        self.trendsignals.extend(trendsignals or [])

        # loop and add invoke all signals' add_signal method (add them to df)
        for signal in signals + trendsignals:
            df = df.pipe(signal.add_signal)

        return df
    
    def trend_changed(self, i, side):
        # check each trend signal
        # ema_trend, macd_trend, ema_slope
        # dont check on first candle of trade?
        # not currently used
        return True in [s.trend_changed(i=i, side=side) for s in self.trendsignals]
            
class Position():
    def __init__(self, contracts=0):
        self.contracts = contracts
        self.orders = []

    def side(self):
        return f.side(self.contracts)
    
    def add_order(self, orders):
        if not isinstance(orders, list): orders = [orders]

        for i, o in enumerate(orders):
            if isinstance(o, dict):
                o = self.convert_bitmex(o)
                orders[i] = o
            
            if o.ordtype == 'Market':
                self.contracts += o.contracts
        
        self.orders.extend(orders)

    def set_contracts(self, orders):
        # check bot orders that REDUCE position
        # split orders into upper and lower, loop outwards
        def traverse(orders):
            contracts = self.contracts
            for o in orders:
                if o.reduce:
                    o.contracts = contracts * -1
                
                contracts += o.contracts
            
            return orders

        upper = sorted(filter(lambda x: x.checkside == -1, orders), key=lambda x: x.price)
        lower = sorted(filter(lambda x: x.checkside == 1, orders), key=lambda x: x.price, reverse=True)
                
        orders=[]
        orders.extend(traverse(upper))
        orders.extend(traverse(lower))

        return orders

    def final_orders(self):
        # Split orders into market and non market, process, then recombine
        ordtype = 'Market'
        orders = [o for o in self.orders if o.ordtype == ordtype]

        nonmarket = [o for o in self.orders if o.ordtype != ordtype]
        orders.extend(self.set_contracts(orders=nonmarket))
        
        orders = list(filter(lambda x: x.manual == False and x.contracts != 0, orders))

        return orders

    def convert_bitmex(self, o):
        # Covert Bitmex manual orders to Order()
        # TODO: This doesn't preserve all attributes (eg orderID), but creates a new order from scratch.
        price = o['stopPx'] if o['ordType'] == 'Stop' else o['price']

        return Order(
            symbol=o['symbol'],
            price=price,
            contracts=o['contracts'],
            ordtype=o['ordType'],
            name=o['name'])

class Strategy():
    def __init__(self, weight=1, lev=5, slippage=0.02, **kw):
        self.i = 1
        self.status = 0
        self.weight = weight
        self.entryprice = 0
        self.exitprice = 0
        self.maxspread = 0.1
        self.slippage = slippage
        self.lev = lev
        self.unfilledtrades = 0
        self.timeout = float('inf')
        
        self.trades = []
        self.trade = None
        self.sym = None
        self.cdl =  None
        self.conf = Confidence()
            
    def tradecount(self):
        return len(self.trades)

    def last_trade(self):
        return self.trades[self.tradecount() - 1]

    def get_trade(self, i):
        numtrades = self.tradecount()
        if i > numtrades: i = numtrades
        return self.trades[i - 1]

    def good_trades(self):
        count = 0
        for t in self.trades:
            if t.is_good():
                count += 1
        return count

    def get_side(self):
        status = self.status
        if status == 0:
            return 0
        elif status < 0:
            return - 1
        elif status > 0:
            return 1

    def init_trade(self, trade, side, entryprice, balance=None, temp=False, conf=None):
        if balance is None:
            balance = self.sym.account.get_balance()

        contracts = f.get_contracts(balance, self.lev, entryprice, side, self.sym.altstatus) * self.weight

        if conf is None:
            conf = self.conf.final(side=side, c=self.cdl)
        
        trade.init(price=entryprice, targetcontracts=contracts, strat=self, conf=conf, side=side, temp=temp)
        
        return trade

    def bad_trades(self):
        return list(filter(lambda x: x.pnlfinal <= 0, self.trades))

    def print_trades(self, maxmin=0, first=float('inf'), last=0, df=None):
        import seaborn as sns
        
        if df is None:
            df = self.result(first=first, last=last)
        style = df.style.hide_index()

        figs = self.sym.decimalfigs
        price_format = '{:,.' + str(figs) + 'f}'

        cmap = sns.diverging_palette(10, 240, sep=10, n=20, center='dark', as_cmap=True)
        style.background_gradient(cmap=cmap, subset=['Pnl'], vmin=-0.1, vmax=0.1)
        style.background_gradient(cmap=cmap, subset=['PnlAcct'], vmin=-0.3, vmax=0.3)

        style.format({'Timestamp': '{:%Y-%m-%d %H}',
                    'Contracts': '{:,}',
                    'Entry': price_format,
                    'Exit': price_format,
                    'Conf': '{:.3f}',
                    'Pnl': '{:.2%}',
                    'PnlAcct': '{:.2%}',
                    'Bal': '{:.2f}'})
        display(style)

    def result(self, first=float('inf'), last=0):
        data = []
        trades = self.trades
        cols = ['N', 'Timestamp', 'Sts', 'Dur', 'Entry', 'Exit',  'Contracts', 'Conf', 'Pnl', 'PnlAcct', 'Bal'] #'Market',

        for t in trades[last * -1: min(first, len(trades))]:
            data.append([
                t.tradenum,
                t.candles[0].Index,
                t.status,
                t.duration(),
                t.entryprice,
                t.exitprice,
                # t.exit_order().ordtype_str(),
                t.filledcontracts,
                t.conf,
                t.pnlfinal,
                t.pnl_acct(),
                t.exitbalance])
        
        return pd.DataFrame.from_records(data=data, columns=cols) \
            .assign(profitable=lambda x: x.Pnl > 0)

class Trade():
    def __init__(self):
        self.candles = []
        self.orders = []
        self.active = True
        self.filledcontracts = 0    
        self.contracts = 0
        self.entryprice = 0
        self.exitprice = 0
        self.pnlfinal = 0
        self.maxpnl = 0
        self.iType = 1
        self.sym = None
        self.strat = None
        self.exitbalance = 0
        self.exitcontracts = 0
        self.partial = False
        self.timedout = False
        self.trend_changed = False
        self.stopped = False

    def init(self, price, targetcontracts, strat, entryprice=0, conf=1, entryrow=0, side=None, temp=False):
        self.entrytarget = price
        self.entryprice = entryprice
        self.targetcontracts = int(targetcontracts)
        self.strat = strat
        self.sym = strat.sym
        self.conf = round(conf, 3)
        self.tradenum = strat.tradecount()
        self.timeout = strat.timeout
        self.cdl = self.sym.cdl
        self.entrybalance = self.sym.account.get_balance()
        self.i_enter = self.strat.i

        if side is None:
            self.status = strat.status
            self.side = strat.get_side() # TODO: sketch
        else:
            self.status = side
            self.side = side
        
        if not temp:
            self.strat.trades.append(self)

        self.enter()

    def exit_trade(self):
        # if not exitprice is None:
        #     self.exitprice = exitprice

        self.strat.status = 0
        self.pnlfinal = f.get_pnl(self.side, self.entryprice, self.exitprice)
        self.exitbalance = self.sym.account.get_balance()
        self.i_exit = self.strat.i
        self.active = False

    def close_order(self, price, contracts):
        
        if contracts == 0: return

        self.exitprice = (self.exitprice * self.exitcontracts + price * contracts) / (self.exitcontracts + contracts)

        if self.entryprice == 0:
            raise ValueError('entry price cant be 0!')

        self.sym.account.modify(
            xbt=f.get_pnl_xbt(contracts * -1, self.entryprice, price, self.sym.altstatus),
            timestamp=self.cdl.Index)
        
        self.exitcontracts += contracts
        self.contracts += contracts

    def close_position(self):
        closeprice = self.sym.cdl.Open
        self.close_order(price=closeprice, contracts=self.contracts * -1)
        self.deactivate_orders(closeall=True) # this is only Trade_Chop

    def get_candle(self, i):
        return self.candles[i - 1]

    def add_candle(self, cdl):
        self.candles.append(cdl)
        self.cdl = cdl
    
    def duration(self):
        offset = -1 if self.partial else 0
        return len(self.candles) + offset

    def pnl_acct(self):
        if self.exitbalance == 0:
            return 0
        else:
            return ((self.exitbalance - self.entrybalance) / self.entrybalance)

    def pnl_xbt(self):
        # not used
        return f.get_pnl_xbt(contracts=self.filledcontracts,
                            entryprice=self.entryprice,
                            exitprice=self.exitprice,
                            isaltcoin=self.sym.altstatus)

    def pnl_current(self, c=None):
        if c is None: c = self.get_candle(self.duration())
        return f.get_pnl(self.side, self.entryprice, c.Close)

    def pnl_maxmin(self, maxmin, firstonly=False):
        return f.get_pnl(self.side, self.entryprice, self.extremum(self.side * maxmin, firstonly))

    def is_good(self):
        ans = True if self.pnlfinal > 0 else False
        return ans

    def is_stopped(self):
        # NOTE this might be old/not used now that individual orders are used
        ans = True if self.pnl_maxmin(-1) < self.strat.stoppercent else False
        return ans

    def exit_date(self):
        return self.candles[self.duration()].timestamp
    
    def rescale_orders(self, balance):
        # need to fix 'orders' for trade_chop
        for order in self.orders:
            order.rescale_contracts(balance=balance, conf=self.conf)
    
    def extremum(self, highlow, firstonly=False):
        
        # entry candle
        c = self.candles[0]
        with f.Switch(self.status * highlow) as case:
            if case(1, -2):
                if highlow == 1:
                    ext = c.High
                elif highlow == -1:
                    ext = c.Low
            elif case(-1, 2):
                ext = self.entryprice

        if firstonly: return ext

        # middle candles
        for i in range(1, self.duration() - 2):
            c = self.candles[i]
            if highlow == 1:
                if c.High > ext: ext = c.High
            elif highlow == -1:
                if c.Low < ext: ext = c.Low

        # exit candle
        c = self.candles[self.duration() - 1]
        with f.Switch(self.status * highlow) as case:
            if case(-1, 2):
                fExt = self.exitprice
            elif case(1, -2):
                if highlow == 1:
                    fExt = c.High
                elif highlow == -1:
                    fExt = c.Low
        
        ext = fExt if (fExt - ext) * highlow > 0 else ext

        return ext

    def all_orders(self):
        return self.orders

    def exit_order(self):
        return list(filter(lambda x: 'close' in x.name, self.orders))[0]

    def df(self):
        return self.sym.df.iloc[self.i_enter:self.i_exit]

    def print_orders(self, orders=None):
        if orders is None:
            orders = self.all_orders()

        data = []
        cols = ['IDX', 'Type', 'Name', 'Side', 'Price', 'PxOriginal', 'Cont', 'Active', 'Cancelled', 'Filled', 'Filltype']

        for o in orders:
            ordtype_bot = o.ordarray.letter() if not o.ordarray is None else o.ordtype_bot 

            data.append([
                o.index,
                ordtype_bot,
                o.name,
                o.side,
                o.price,
                o.pxoriginal,
                o.contracts,
                o.active,
                o.cancelled,
                o.filled,
                o.ordtype_str()])

        df = pd.DataFrame(data=data, columns=cols)
        display(df)

class Order():
    def __init__(self, contracts, ordtype, price=None, side=None, ordtype_bot=None, reduce=False, trade=None, sym=None,  activate=False, index=0, symbol=None, name='', execinst=[], ordarray=None,):

        self.trade = trade
        self.sym = sym

        self.symbol = symbol
        self.name = name.lower()
        self.orderID = ''
        self.ordtype = ordtype
        self.reduce = reduce

        self.active = activate
        self.delaytime = None
        self.filled = False
        self.marketfilled = False
        self.cancelled = False
        self.filledtime = None
        
        self.livedata = []

        # EXEC INST
        if not isinstance(execinst, list): execinst = [execinst]
        self.execinst = []
        self.execinst.extend(execinst)

        if self.ordtype == 'Limit': self.execinst.append('ParticipateDoNotInitiate')
        if 'stop' in self.name: self.execinst.append('IndexPrice')
        if 'close' in self.name: self.execinst.append('Close')

        self.isstop = True if self.ordtype == 'Stop' else False     

        self.ordarray = ordarray
        if not ordarray is None:
            self.index = index
            self.activenext = False # only used in ordarray, superceeded with delaytime
            self.trade = ordarray.trade
        else:
            self.index = pd.NA

        if not self.trade is None:
            self.manual = False
            self.sym = self.trade.strat.sym
            self.slippage = self.trade.strat.slippage
            self.trade.orders.append(self)
        else:
            self.manual = True
            if self.name == '': self.name = 'manual'
        
        # live trading
        if not self.sym is None:
            self.decimalfigs = self.sym.decimalfigs
            self.symbol = symbol
            self.symbolbitmex = self.sym.symbolbitmex
        else:
            self.decimalfigs = 0
            self.symbolbitmex = self.symbol
            if self.symbol is None: raise NameError('Symbol required!')
        
        self.decimaldouble = float('1e-{}'.format(self.decimalfigs))

        """
        # enterexit: 1,2 happen in same direction, 3,4 opposite (used to check if filled)
        # addsubtract: 1,4 add contracts to trade, 2,3 remove, relative to side
        # 1 - LimitOpen
        # 2 - StopClose
        # 3 - LimitClose
        # 4 - StopOpen
        # 5 - MarketOpen - only used for addsubtract
        # 6 - MarketClose"""
        if ordtype_bot:
            self.ordtype_bot = ordtype_bot
            self.enterexit = -1 if ordtype_bot in (1, 2) else 1
            self.addsubtract = -1 if ordtype_bot in (2, 3, 6) else 1

            if not self.trade is None:
                self.direction = self.trade.side * self.enterexit          
        
        # If Side is passed explicitly > force it, else get from contracts
        if not side is None:
            self.side = side
            self.contracts = abs(contracts) * side
        else:
            self.side = 1 if contracts > 0 else -1
            self.contracts = contracts
        
        self.checkside = self.side * -1 if self.isstop else self.side

        if price is None and not self.ordtype == 'Market':
            raise ValueError('Price cannot be None!')

        self.price = self.final_price(price)
        self.pxoriginal = self.price
        self.set_key()

    def final_price(self, price=None):
        if self.ordtype == 'Market':
            return None
        elif self.manual:
            return price
        else:
            if price is None:
                price = self.price
            return round(round(price, self.decimalfigs) + self.decimaldouble * self.side * -1, self.decimalfigs)
    
    def ordtype_str(self):
        # v sketch
        if self.filled:
            ans = 'L' if not self.marketfilled else 'M'
        else:
            ans = pd.NA

        return ans

    def set_name(self, name):
        self.name = name
        self.set_key()

    def set_key(self):
        side = self.side if self.trade is None else self.trade.side

        self.key = f.key(self.symbolbitmex, self.name, side, self.ordtype)
        self.clOrdID = '{}-{}'.format(self.key, int(time()))        

    def stoppx(self):
        return self.price * (1 + self.slippage * self.side)

    def out_of_range(self, c):
        # not used
        pnl = f.get_pnl(entryprice=self.price,
                        exitprice=c.Close,
                        side=self.side)

        ans = True if pnl > 0.903 else False
        return ans

    def set_price(self, price):
        if not self.filled:
            self.price = price
                                
    def check_stop_price(self):
        """Use price if Limit/Market, else use slippage price"""
        price = self.price if not self.isstop else self.stoppx()
        return price
                                
    def check(self, c):
        checkprice = c.High if self.direction == 1 else c.Low
        
        if self.direction * (self.price - checkprice) <= 0:
            self.fill(c=c)
            
    def open_(self, price):
        t = self.trade
        contracts = self.contracts

        if contracts == 0: return

        t.entryprice = (t.entryprice * t.contracts + price * contracts) / (t.contracts + contracts)
        t.contracts += contracts            
            
    def close(self, price):
        self.trade.close_order(price=price, contracts=self.contracts)
            
    def fill(self, c, price=None):
        self.filled = True
        self.filledtime = c.Index

        # market filling
        if not price is None:
            self.price = price
            self.marketfilled = True
        
        price = self.check_stop_price()

        self.open_(price=price) if self.addsubtract == 1 else self.close(price=price)
            
    def is_active(self, c):
        if not self.delaytime is None:
            active = True if self.active and c.Index >= self.delaytime else False
        else:
            active = self.active

        return active        
            
    def activate(self, c, delay=0):
        self.delaytime = c.Index + delta(hours=delay)
        self.active = True
            
    def print_self(self):
        print(
            self.index,
            self.side,
            self.price,
            self.contracts,
            self.active,
            self.cancelled,
            self.filled)

    def cancel(self):
        self.active = False
        self.cancelled = True
        self.filledtime = self.sym.cdl.Index
        if not self.ordarray is None:
            self.ordarray.openorders -= 1

    def rescale_contracts(self, balance, conf=1):
        self.contracts = int(conf * f.get_contracts(
                        xbt=balance,
                        leverage=self.trade.strat.lev,
                        entryprice=self.price,
                        side=self.side,
                        isaltcoin=self.sym.altstatus))

    def intake_live_data(self, livedata):
        self.livedata = livedata
        self.orderID = livedata['orderID']

    def append_execinst(self, m):
        if self.execinst:
            if isinstance(self.execinst, list):
                m['execInst'] = ','.join(self.execinst)
            else:
                m['execInst'] = self.execinst

        return m

    def amend_order(self):
        m = {}
        m['orderID'] = self.orderID
        m['symbol'] = self.sym.symbolbitmex
        m['orderQty'] = self.contracts
        m = self.append_execinst(m)

        with f.Switch(self.ordtype) as case:
            if case('Limit'):
                m['price'] = self.price
            elif case('Stop'):
                m['stopPx'] = self.price
        
        return m

    def to_dict(self):
        m = self.new_order()
        m['contracts'] = self.contracts
        m['side'] = self.side
        m['name'] = self.name
        return m
    
    def new_order(self):
        m = {}
        m['symbol'] = self.symbolbitmex
        m['orderQty'] = self.contracts
        m['clOrdID'] = self.clOrdID
        m['ordType'] = self.ordtype
        m = self.append_execinst(m)
        
        with f.Switch(self.ordtype) as case:
            if case('Limit'):
                m['price'] = self.price
            elif case('Stop'):
                m['stopPx'] = self.price
            elif case('Market'):
                m['ordType'] = self.ordtype
        
        return m

class OrdArray():
    
    def getFraction(self, n):
        if n == 0:
            return 1 / 6
        elif n == 1:
            return 1 / 4.5
        elif n == 2:
            return 1 / 3.6
        elif n == 3:
            return 1 / 3

    def getOrderPrice(self, n):
        if not self.outerprice is None:
            n += 1
        
        price = self.anchorprice * (1 + self.orderspread * n * self.trade.status * self.enterexit) # TODO: this won't work, enterexit moved to order
        
        return round(price, self.decimalfigs)
    
    def __init__(self, ordtype_bot, anchorprice, orderspread, trade, activate=False, outerprice=None):
        self.ordtype_bot = ordtype_bot
        self.anchorprice = anchorprice
        self.orderspread = orderspread
        self.trade = trade
        self.outerprice = outerprice
        
        if not outerprice is None:
            self.pricerange = abs(self.outerprice - self.anchorprice)
            self.orderspread = (self.pricerange / (trade.numorders + 1))  / anchorprice

        self.decimalfigs = self.trade.strat.sym.decimalfigs

        self.orders = []
        self.active = True
        self.side = self.trade.side * self.addsubtract

        self.filled = False
        self.filledorders = 0
        self.filledcontracts = 0
        self.numorders = self.trade.numorders
        self.openorders = self.numorders

        self.maxprice = anchorprice * (1 + self.direction * ((self.numorders - 1) * orderspread))

        # init and add all orders to self (ordArray)
        modi = 'lwr' if self.trade.side == 1 else 'upr'

        with f.Switch(self.ordtype_bot) as case:
            if case(1):
                ordtype = 'Limit'
            elif case(2):
                ordtype = 'Stop'
            elif case(3):
                ordtype = 'Limit'

        for i in range(self.numorders):
            price = self.getOrderPrice(i)
            contracts = int(round(self.getFraction(i) * self.trade.targetcontracts * self.addsubtract, 0))

            order = Order(
                        price=price,
                        side=self.side,
                        contracts=contracts,
                        ordarray=self,
                        activate=activate,
                        index=i,
                        symbol=self.trade.strat.sym.symbol,
                        name='{}{}{}'.format(self.letter(), i + 1, modi),
                        ordtype=ordtype,
                        sym=self.trade.strat.sym)

            self.orders.append(order)
    
    def check_orders(self, c):
        if not self.active:
            return
        
        if self.ordtype_bot > 1 and self.trade.contracts == 0:
            return
        
        for i, order in enumerate(self.orders):
            
            if order.active and not order.filled:
                order.check(c)
                if order.filled:
                    if self.ordtype_bot == 1:
                        self.trade.stops.orders[i].active = True
                        self.trade.takeprofits.orders[i].activenext = True
                    elif self.ordtype_bot == 2:
                        self.trade.takeprofits.orders[i].cancel()
                    elif self.ordtype_bot == 3:
                        self.trade.stops.orders[i].cancel()

                    self.filledorders += 1
                    self.filledcontracts += order.contracts
            elif order.activenext and not order.cancelled:
                # delay for 1 period
                order.activenext = False
                order.active = True
        
        if self.filledorders == self.openorders:
            self.filled = True
            self.active = False

    def letter(self):
        with f.Switch(self.ordtype_bot) as case:
            if case(1):
                return 'O'
            elif case(2):
                return 'S'
            elif case(3):
                return 'T'

    def strfilled(self):
        return '{}{}{}'.format(self.letter(), self.filledorders, self.openorders)
    
    def getUnfilledOrders(self, targetcontracts=None, actualcontracts=0):
        lst = []
        # print('actualcontracts: {}'.format(actualcontracts))
        for i, order in enumerate(self.orders):
            
            # rescale to contracts to reflect actual user balance
            if not targetcontracts is None:
                order.contracts = int(round(self.getFraction(order.index) * targetcontracts * self.addsubtract, 0))

            if not (order.cancelled or order.filled):
                
                if self.ordtype_bot == 1:
                    # order
                    lst.append(order)
                else:
                    # stops - check if matching order NOT filled
                    # print(self.trade.orders.orders[i].filled)
                    if not self.trade.orders.orders[i].filled:
                        # good, stops should be active
                        if self.ordtype_bot == 2:
                            lst.append(order)
                    else:
                        # Order SHOULD be filled, check it
                        # loop from max filled order to current order, check if we have enough contracts
                        ordarray = self.trade.orders
                        maxfilled = ordarray.filledorders
                        # print('i: {}, maxfilled: {}'.format(i, maxfilled))
                        remainingcontracts = actualcontracts
                        for y in range(maxfilled - 1, i, -1):
                            # print(y, remainingcontracts, ordarray.orders[y].contracts)
                            remainingcontracts -= ordarray.orders[y].contracts

                        # print('remainingfinal: {}, order.contr: {}, side: {}'.format(remainingcontracts, order.contracts, ordarray.side))
                        if (remainingcontracts - order.contracts * -1) * ordarray.side >= 0:
                            lst.append(order)
                            # also check to fill the last order no matter what??
        return lst

    def print_orders(self):
        for order in self.orders:
            order.print_self()

class Txn():
    def __init__(self):
        self.amount = 0
        self.timestamp = None
        self.balance_pre = 0
        self.balance_post = 0
        self.percentchange = 0

    def printTxn(self):
        pass

class Candle():
    def __init__(self, row):
        self.row = row

    def dHC(self):
        c = self.row
        return c.High - c.Close
    
    def dLC(self):
        c = self.row
        return c.Low - c.Close
    
    def percentOCd(self):
        c = self.row
        return (c.Close - c.Open) / c.Open

    def percentOC(self):
        return f.percent(self.percentOCd())
            
    def size(self):
        c = self.row
        return abs(c.High - c.Low)

    def side(self):
        c = self.row
        ans = 1 if c.Close > c.Open else -1
        return ans
    
    def tailpct(self, side):
        return self.tailsize(side=side) / self.size()
    
    def tailsize(self, side):
        # 1 = upper, -1 = lower
        # tailsize depends on side of candle
        c = self.row

        if side == 1:
            return c.High - max(c.Close, c.Open)
        elif side == -1:
            return min(c.Close, c.Open) - c.Low

    def getmax(self, side):
        c = self.row
        if side == 1:
            return c.High
        else:
            return c.Low
