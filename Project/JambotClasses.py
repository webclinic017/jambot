import os
from datetime import datetime as date
from datetime import timedelta as delta
from enum import Enum
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

import Functions as f
import LiveTrading as live

try:
    from IPython.display import display
except ModuleNotFoundError:
    pass



# BACKTEST
class Account():
    
    def __init__(self):
        self.balance = 1
        self.max = 0
        self.min = 1
        self.txns = list()

    def getBalance(self):
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
        txn.acctbalance = self.balance
        txn.percentchange = round((xbt / self.balance), 3)

        self.txns.append(txn)
        self.balance = self.balance + xbt
        if self.balance > self.max: self.max = self.balance
        if self.balance < self.min: self.min = self.balance    

    def getPeriodNum(self, timestamp, period='month'):
        timestamp = f.checkDate(timestamp)
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
            val = txn.acctbalance
            if val > max_seen:
                max_seen = val
                txmax_seen = txn

            curdraw = 1 - val / max_seen
            if curdraw > drawdown:
                drawdown = curdraw
                txlow = txn
                txhigh = txmax_seen
        
        drawdates = '{:.2f} ({}) - {:.2f} ({})'.format(
            txhigh.acctbalance,
            date.strftime(txhigh.timestamp, f.TimeFormat()),
            txlow.acctbalance,
            date.strftime(txlow.timestamp, f.TimeFormat()))
        
        return drawdown * -1, drawdates
    
    def getPercentChange(self, balance, change):
        return f.percent(change / balance)

    def printsummary(self, period='month'):
        # TODO: replace columnar with dataframe
        from columnar import columnar
        data = []
        headers = ['Period', 'AcctBalance', 'Change', 'PercentChange']

        periodnum = self.getPeriodNum(self.txns[0].timestamp, period)
        prevTxn = None
        prevBalance = 1
        change = 0.0

        for t in self.txns:     
            if self.getPeriodNum(t.timestamp, period) != periodnum:
                if prevTxn is None: prevTxn = t
                data.append([
                    periodnum,
                    '{:.{prec}f}'.format(t.acctbalance, prec=3),
                    round(change, 3),
                    self.getPercentChange(prevBalance, change)
                ])
                prevTxn = t
                prevBalance = t.acctbalance
                change = 0
                periodnum = self.getPeriodNum(t.timestamp, period)
            
            change = change + t.amount

        table = columnar(data, headers, no_borders=True, justify='r')
        print(table)

    def plotbalance(self, logy=False, title=None):
        m = {}
        m['timestamp'] = [t.timestamp for t in self.txns]
        m['balance'] = [t.acctbalance for t in self.txns]
        df = pd.DataFrame.from_dict(m)
        df.plot(kind='line', x='timestamp', y='balance', logy=logy, linewidth=0.75, color='cyan', title=title)

    def printtxns(self):
        # TODO: remove columnar
        from columnar import columnar
        data = []
        headers = ['Date', 'AcctBalance', 'Amount', 'PercentChange']
        for t in self.txns:
            data.append([
                '{:%Y-%m-%d %H}'.format(t.timestamp),
                '{:.{prec}f}'.format(t.acctbalance, prec=3),
                '{:.{prec}f}'.format(t.amount, prec=2),
                f.percent(t.percentchange)
            ])
        table = columnar(data, headers, no_borders=True, justify='r')
        print(table)

    def getDf(self):
        df = pd.DataFrame(columns=['Timestamp', 'Balance', 'PercentChange'])
        for i, t in enumerate(self.txns):
            df.loc[i] = [t.timestamp, t.acctbalance, t.percentchange]
        return df

class Backtest():
    def __init__(self, symbol, startdate=None, strats=[], daterange=365, df=None, row=None, account=None, partial=False, u=None):

        if not isinstance(strats, list): strats = [strats]

        if row is None:
            dfsym = pd.read_csv(Path(f.curdir()) / 'symbols.csv')
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
        self.startdate = f.checkDate(startdate)
        self.strats = strats

        # actual backtest, not just admin info
        if not startdate is None:
            if account == None:
                self.account = Account()

            self.i = 1
            self.candles = []
            
            if df is None:
                self.df = f.getDataFrame(symbol=symbol, startdate=startdate, daterange=daterange)
            else:
                self.df = df

            if partial:
                if u is None: u = live.User()
                self.df = u.appendpartial(df)
           
            self.startrow = self.df.loc[self.df['Timestamp'] == pd.Timestamp(self.startdate)].index[0]

            for strat in self.strats:
                strat.init(sym=self)

    def initCandle(self, c):
        self.cdl = c
        self.candles.append(c)

    def decidefull(self):
        length = len(self.df)
        for c in self.df.itertuples():
            self.initCandle(c=c)
            
            self.i = c.Index
            i = self.i
            if not i < self.startrow:
                if i == length: return

                for strat in self.strats:
                    strat.decide(c)
        
        if self.partial:
            self.strats[0].trades[-1].partial = True

    def printfinal(self):
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
            'Goodtrades': ['{}/{}/{}'.format(strat.goodtrades(), strat.tradecount(), strat.unfilledtrades)]}

        return pd.DataFrame(data=data)

    def writecsv(self):
        self.df.to_csv('dfout.csv')
        self.account.getDf().to_csv('df2out.csv')

    def expectedOrders(self):
        # don't use this yet.. maybe when we have combined strats?
        
        expected = []
        for strat in self.strats:
            for order in strat.finalOrders():
                expected.append(order)
        
        return expected


# CONFIDENCE
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

    def addsignal(self, signals=[], trendsignals=[]):
        self.signals.extend(signals)
        self.trendsignals.extend(trendsignals)
    
    def trendchanged(self, i, side):
        # check each trend signal
        # ema_trend, macd_trend, ema_slope
        # dont check on first candle of trade?
        # not currently used
        return True in [s.trendchanged(i=i, side=side) for s in self.trendsignals]
            
class Signal():
    def __init__(self, df, weight=1):
        self.df = df
        self.weight = weight
        self.trendseries = None
    
    def trendchanged(self, i, side):
        tseries = self.trendseries
        tnow = tseries.iloc[i]
        tprev = tseries.iloc[i - 1]
        return not tnow == side and not tnow == tprev

class Signal_MACD(Signal):
    def __init__(self, df, weight=1, fast=50, slow=200, smooth=50):
        super().__init__(df=df, weight=weight)
        # init macd series
        self.name = 'macd'
        f.addEma(df=df, p=fast)
        f.addEma(df=df, p=slow)

        df['macd'] = df[f'ema{fast}'] - df[f'ema{slow}']
        df['macd_signal'] = df.macd.ewm(span=smooth, min_periods=smooth).mean()
        df['macd_diff'] =  df.macd - df.macd_signal
        df['macd_trend'] = np.where(df.macd_diff > 0, 1, -1)

        self.trendseries = df.macd_trend

    def final(self, side, c):
        if side * c.macd_trend == 1:
            conf = 1.25
        else:
            conf = 0.5

        return conf * self.weight        

class Signal_EMA(Signal):
    def __init__(self, df, weight=1, fast=50, slow=200):
        super().__init__(df=df, weight=weight)
        self.name = 'ema'
        f.addEma(df=df, p=fast)
        f.addEma(df=df, p=slow)
        colfast, colslow = f'ema{fast}', f'ema{slow}'

        df['emaspread'] = round((df[colfast] - df[colslow]) / ((df[colfast] + df[colslow]) / 2) , 6)
        df['ema_trend'] = np.where(df[colfast] > df[colslow], 1, -1)

        c = self.getC(maxspread=0.1)
        df['ema_conf'] = self.emaExp(x=df.emaspread, c=c)

        self.trendseries = df.ema_trend

    def final(self, side, c):
        temp_conf = abs(c.ema_conf)

        if side * c.ema_trend == 1:
            conf = 1.5 - temp_conf * 2
        else:
            conf = 0.5 + temp_conf * 2
        
        return conf * self.weight
            
    def getC(self, maxspread):
        m = -2.9
        b = 0.135
        return round(m * maxspread + b, 2)

    def emaExp(self, x, c):
        side = np.where(x >= 0, 1, -1)
        x = abs(x)
        
        aLim = 2
        a = -1000
        b = 3
        d = -3
        g = 1.7

        y = side * (a * x ** b + d * x ** g) / (aLim * a * x ** b + aLim * d * x ** g + c)

        return round(y, 6)

class Signal_EMA_Slope(Signal):
    def __init__(self, df, weight=1, p=50, slope=5):
        super().__init__(df=df, weight=weight)
        self.name = 'ema_Slope'
        f.addEma(df=df, p=p)
        df['ema_slope'] = np.where(np.roll(df['ema{}'.format(p)], slope, axis=0) < df['ema{}'.format(p)], 1, -1)
        df.loc[:p + slope, 'ema_slope'] = np.nan

        self.trendseries = df.ema_slope

    def final(self, side, c):
        if side * c.ema_slope == 1:
            conf = 1.5
        else:
            conf = 0.5

        return conf * self.weight     

class Signal_Volatility(Signal):
    def __init__(self, df, weight=1, norm=(0.004,0.024)):
        super().__init__(df=df, weight=weight)
        self.name = 'volatility'

        df['maxhigh'] = df.High.rolling(48).max()
        df['minlow'] = df.Low.rolling(48).min()
        df['spread'] = abs(df.maxhigh - df.minlow) / df[['maxhigh', 'minlow']].mean(axis=1)

        df['emavty'] = df.spread.ewm(span=60, min_periods=60).mean()
        df['smavty'] = df.spread.rolling(300).mean()
        df['norm_ema'] = np.interp(df.emavty, (0, 0.25), (norm[0], norm[1]))
        df['norm_sma'] = np.interp(df.smavty, (0, 0.25), (norm[0], norm[1]))
        # df['normtp'] = np.interp(df.smavty, (0, 0.4), (0.3, 3)) # only Strat_Chop

        df.drop(columns=['maxhigh', 'minlow'], inplace=True)

    def final(self, c):
        # return self.df.norm_ema[i]
        return c.norm_ema

class Signal_Trend(Signal):
    def __init__(self, df, signals, speed, offset=1):
        super().__init__(df=df)
        # accept 1 to n series of trend signals, eg 1 or -1
        # sum signals > positive = 1, negative = -1, neutral = 0
        # df['temp'] = np.sum(signals, axis=0)
        # df['trend'] = np.where(df.temp == 0, 0, np.where(df.temp > 0, 1, -1))
        # ^didn't work, just use ema_trend for now
        df['temp'] = np.nan
        df['trend'] = df.ema_trend

        # set trade high/low in period prices
        against, wth, neutral = speed[0], speed[1], int(np.average(speed))

        # max highs
        df['mhw'] = df.High.rolling(wth).max().shift(offset)
        df['mha'] = df.High.rolling(against).max().shift(offset)
        df['mhn'] = df.High.rolling(neutral).max().shift(offset)

        # min lows
        df['mla'] = df.Low.rolling(wth).min().shift(offset)
        df['mlw'] = df.Low.rolling(against).min().shift(offset)
        df['mln'] = df.Low.rolling(neutral).min().shift(offset)

        df['pxhigh'] = np.where(df.trend == 0, df.mhn, np.where(df.trend == 1, df.mha, df.mhw))
        df['pxlow'] = np.where(df.trend == 0, df.mln, np.where(df.trend == -1, df.mlw, df.mla))
        
        df.drop(columns=['mha', 'mhw', 'mla', 'mlw', 'mhn', 'mln', 'temp'], inplace=True)

    def final(self, c):
        return c.trend


# STRAT
class Strategy():
    def __init__(self, weight=1, lev=5):
        self.i = 1
        self.status = 0
        self.weight = weight
        self.entryprice = 0
        self.exitprice = 0
        self.maxspread = 0.1
        self.slippage = 0.002
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

    def lasttrade(self):
        return self.trades[self.tradecount() - 1]

    def getTrade(self, i):
        numtrades = self.tradecount()
        if i > numtrades: i = numtrades
        return self.trades[i - 1]

    def goodtrades(self):
        count = 0
        for t in self.trades:
            if t.isgood():
                count += 1
        return count

    def getSide(self):
        status = self.status
        if status == 0:
            return 0
        elif status < 0:
            return - 1
        elif status > 0:
            return 1

    def checkfinalorders(self, finalorders):
        for i, order in enumerate(finalorders):
            if order.contracts == 0:
                del finalorders[i]
        return finalorders

    def inittrade(self, trade, side, entryprice, balance=None, temp=False, conf=None):
        if balance is None:
            balance = self.sym.account.getBalance()

        contracts = f.getContracts(balance, self.lev, entryprice, side, self.sym.altstatus) * self.weight

        if conf is None:
            conf = self.conf.final(side=side, c=self.cdl)
        
        trade.init(price=entryprice, targetcontracts=contracts, strat=self, conf=conf, side=side, temp=temp)
        
        return trade

    def badtrades(self):
        return list(filter(lambda x: x.pnlfinal <= 0, self.trades))

    def printtrades(self, maxmin=0, first=float('inf'), last=0, df=None):
        import seaborn as sns
        
        if df is None:
            df = self.result(first=first, last=last)
        style = df.style.hide_index()

        figs = self.sym.decimalfigs
        priceformat = '{:,.' + str(figs) + 'f}'

        cmap = sns.diverging_palette(10, 240, sep=10, n=20, center='dark', as_cmap=True)
        style.background_gradient(cmap=cmap, subset=['Pnl'], vmin=-0.1, vmax=0.1)
        style.background_gradient(cmap=cmap, subset=['PnlAcct'], vmin=-0.3, vmax=0.3)

        style.format({'Timestamp': '{:%Y-%m-%d %H}',
                    'Contracts': '{:,}',
                    'Entry': priceformat,
                    'Exit': priceformat,
                    'Conf': '{:.3f}',
                    'Pnl': '{:.2%}',
                    'PnlAcct': '{:.2%}',
                    'Bal': '{:.2f}'})
        display(style)

    def result(self, first=float('inf'), last=0):
        data = []
        trades = self.trades
        cols = ['N', 'Timestamp', 'Sts', 'Dur', 'Entry', 'Exit', 'Market', 'Contracts', 'Conf', 'Pnl', 'PnlAcct', 'Bal']

        for t in trades[last * -1: min(first, len(trades))]:
            data.append([
                t.tradenum,
                t.candles[0].Timestamp,
                t.status,
                t.duration(),
                t.entryprice,
                t.exitprice,
                t.exitorder().ordtype_str(),
                t.filledcontracts,
                t.conf,
                t.pnlfinal,
                t.pnlacct(),
                t.exitbalance])
        
        return pd.DataFrame.from_records(data=data, columns=cols)

class Strat_TrendRev(Strategy):
    # try variable stopout %?

    def __init__(self, speed=(8,8), weight=1, norm=(0.004, 0.024), lev=5):
        super().__init__(weight, lev)
        self.name = 'trendrev'
        self.stoppercent = -0.03
        self.speed = speed
        self.norm = norm
        self.lasthigh, self.lastlow = 0, 0
        self.timeout = 40
        self.slippage = 0

    def init(self, sym):
        self.sym = sym
        df = sym.df
        self.a = self.sym.account
        
        self.vty = Signal_Volatility(df=df, weight=1, norm=self.norm)

        macd = Signal_MACD(df=df, weight=1)
        ema = Signal_EMA(df=df, weight=1)
        emaslope = Signal_EMA_Slope(df=df, weight=1, p=50, slope=5)
        self.conf.addsignal(signals=[macd, ema, emaslope],
                            trendsignals=[ema])

        self.trend = Signal_Trend(df=df, signals=[df.ema_trend, df.ema_slope], speed=self.speed)

    def exittrade(self):
        t = self.trade
        c = self.cdl

        if not t.stopped and t.limitopen.filled and not t.limitclose.filled:
            t.stop.cancel()
            t.limitclose.fill(c=c, price=c.Close)
            self.unfilledtrades += 1
            
        t.exittrade()
        self.trade = None
    
    def entertrade(self, side, entryprice):
        self.trade = self.inittrade(trade=Trade_TrendRev(), side=side, entryprice=entryprice)
        t = self.trade
        t.checkorders(self.sym.cdl)
        if not t.active: self.exittrade()
        
    def decide(self, c):
        self.cdl = c
        self.i = c.Index
        pxhigh, pxlow = c.pxhigh, c.pxlow

        # if we exit a trade and limitclose isnt filled, limitopen may wait and fill in next candles, but current trade gets 0 profit. Need to exit properly.

        # Exit Trade
        if not self.trade is None:
            t = self.trade

            if t.side == 1:
                if c.High > max(pxhigh, self.lasthigh):
                    t.active = False
            else:
                if c.Low < min(pxlow, self.lastlow):
                    t.active = False

            # if c.Timestamp >= date(2020,2,21,19) and c.Timestamp <= date(2020,2,21,20):
            #     print('pxlow: {}, lastlow: {}, active: {}'.format(pxlow, self.lastlow, t.active))

            t.checkorders(c)
            if not t.active: self.exittrade()

            # need to make trade active=False if limitclose is filled?
            # order that candle moves is important
            # TODO: need enter > exit > enter all in same candle

        # if c.Timestamp == date(2020,2,21,20):
        #     print(c.Timestamp)

        # Enter Trade
        if self.trade is None:
            if c.High > pxhigh:
                self.entertrade(-1, pxhigh)
            elif c.Low < pxlow:
                self.entertrade(1, pxlow)

        self.lasthigh, self.lastlow = pxhigh, pxlow

    def finalOrders(self, u, weight):
        lst = []
        symbol = self.sym.symbolbitmex
        balance = u.balance() * weight
        curr_cont = u.getPosition(symbol)['currentQty']
        c = self.cdl

        manualorders = u.getOrders(symbol=symbol, manualonly=True)
        
        # PREV
        # CLOSE - Check if current limitclose is still open with same side
        prevclose = self.trades[-2].limitclose
        if prevclose.marketfilled:
            prevclose_actual = u.getOrderByKey(key=f.key(symbol, 'limitclose', prevclose.side, 3))
            if (not prevclose_actual is None
                and prevclose_actual['side'] == prevclose.side):
                    prevclose.contracts = curr_cont * -1
                    prevclose.setname('marketclose')
                    prevclose.ordtype = 'Market'
                    prevclose.execInst = 'Close'
                    lst.append(prevclose)
                    
                    # subtract from curr_cont when market closing
                    curr_cont += prevclose.contracts
        
        # CURRENT
        t_current = self.trades[-1]
        t_current.rescaleorders(balance=balance)

        # OPEN
        limitopen = t_current.limitopen
        limitopen_actual = u.getOrderByKey(key=f.key(symbol, 'limitopen', limitopen.side, 1))
        
        if limitopen.filled:
            if (limitopen.marketfilled 
            and curr_cont == 0
            and t_current.duration() == 4):
                limitopen.setname('marketbuy') # need diff name, 2 limitbuys possible
                limitopen.ordtype = 'Market'
                limitopen.execinst = ''
                lst.append(limitopen)
                curr_cont += limitopen.contracts
                limitopen_actual['orderQty'] = None # will be cancelled

        else:
            # Only till max 4 candles into trade
            lst.append(limitopen)

        # STOP - depends on either a position OR a limitopen
        stopclose = t_current.stop
        if not stopclose.filled:
            stopclose.contracts = curr_cont * -1
            stopclose.contracts -= f.sum_orders_before(orders=manualorders, checkorder=stopclose)

            # only if limitopen_actual has not been market closed in this period
            if not limitopen_actual['orderQty'] is None:
                stopclose.contracts -= limitopen_actual['orderQty'] * limitopen_actual['side']

            lst.append(stopclose)

        # CLOSE - current
        if not curr_cont == 0:
            limitclose = t_current.limitclose
            limitclose.contracts = curr_cont * -1
            
            if t_current.timedout:
                limitclose.setname('marketclose')
                limitclose.ordtype = 'Market'
                limitclose.execinst = 'Close'
                curr_cont += limitclose.contracts
            else:
                limitclose.contracts -= f.sum_orders_before(orders=manualorders, checkorder=limitclose)

            lst.append(limitclose)

            if stopclose.contracts == 0 or not stopclose in lst:
                f.discord(msg='Error: no stop for current position', channel='err')

        # NEXT - Init next trade to get next limitopen and stop
        px = c.pxhigh if t_current.side == 1 else c.pxlow
        t_next = self.inittrade(side=t_current.side * -1, entryprice=px, balance=balance, temp=True, trade=Trade_TrendRev())
        
        t_next.stop.contracts -= f.sum_orders_before(orders=manualorders, checkorder=t_next.stop)
        lst.append(t_next.limitopen)
        lst.append(t_next.stop)

        return self.checkfinalorders(lst)

class Strat_Trend(Strategy):
    def __init__(self, speed=(18,18), weight=1, lev=5, emaspeed=(50, 200)):
        super().__init__(weight, lev)
        self.name = 'trend'
        self.enteroffset = 0
        self.speed = speed
        self.lasthigh, self.lastlow = 0, 0
        self.emaspeed = emaspeed

    def init(self, sym):
        self.sym = sym
        df = sym.df
        fast, slow = self.emaspeed[0], self.emaspeed[1]
            
        macd = Signal_MACD(df=sym.df, weight=1, fast=fast, slow=slow)
        ema = Signal_EMA(df=sym.df, weight=1, fast=fast, slow=slow)
        # emaslope = Signal_EMA_Slope(df=df, weight=1, p=50, slope=5)
        self.conf.addsignal([macd, ema])
        
        self.trend = Signal_Trend(df=df, signals=[df.ema_trend], speed=self.speed)

    def getRecentWinConf(self):
        # TODO: could probs do this better with list/filter
        recentwin = False
        closedtrades = self.tradecount()
        ctOffset = closedtrades - 1 if closedtrades < 3 else 2
        for y in range(closedtrades, closedtrades - ctOffset - 1, -1):
            if self.getTrade(y).pnlfinal > 0.05:
                recentwin = True
        
        recentwinconf = 0.25 if recentwin else 1
        return recentwinconf

    def getConfidence(self, side):
        conf = self.conf.final(side=side, c=self.cdl)

        recentwinconf = self.getRecentWinConf()
        confidence = recentwinconf if recentwinconf <= 0.5 else conf
        return confidence

    def entertrade(self, side, entryprice):
        self.trade = self.inittrade(trade=Trade_Trend(), side=side, entryprice=entryprice, conf=self.getConfidence(side=side))
        self.trade.checkorders(self.sym.cdl)

    def exittrade(self):
        self.trade.exittrade()

    def decide(self, c):
        self.cdl = c
        self.i = c.Index
        pxhigh, pxlow = c.pxhigh, c.pxlow

        # Exit Trade
        if not self.trade is None:
            self.trade.checkorders(c)

            if self.trade.side == 1:
                if c.Low < pxlow and c.Low < self.lastlow:
                    self.exittrade()                
            else:
                if c.High > pxhigh and c.High > self.lasthigh:
                    self.exittrade()

            if not self.trade.active:
                self.trade = None
        
        # Enter Trade
        if self.trade is None:
            pxhigh *= (1 + self.enteroffset)
            pxlow *= (1 - self.enteroffset)

            if c.High > pxhigh:
                self.entertrade(1, pxhigh)
            elif c.Low < pxlow:
                self.entertrade(-1, pxlow)

        self.lasthigh, self.lastlow = pxhigh, pxlow

    def finalOrders(self, u, weight):
        # should actually pass something at the 'position' level, not user?
        lstOrders = []
        c = self.sym.cdl
        side = self.getSide()
        price = c.trend_low if self.status == 1 else c.trend_high
        
        #TODO: use trade's orders now
        # stopclose
        lstOrders.append(Order(
                    price=price,
                    side=-1 * side,
                    contracts=-1 * u.getPosition(self.sym.symbolbitmex)['currentQty'],
                    symbol=self.sym.symbolbitmex,
                    name='stopclose',
                    ordtype='Stop',
                    sym=self.sym))

        # stopbuy
        contracts = f.getContracts(
                        u.totalbalancewallet * weight,
                        self.lev,
                        price,
                        -1 * side,
                        self.sym.altstatus)

        lstOrders.append(Order(
                    price=price,
                    side=-1 * side,
                    contracts=contracts,
                    symbol=self.sym.symbolbitmex,
                    name='stopbuy',
                    ordtype='Stop',
                    sym=self.sym))

        return self.checkfinalorders(lstOrders)
        
class Strat_TrendClose(Strat_Trend):
    def __init__(self, speed=(18,18), emaspeed=(200, 800)):
        super().__init__(speed=speed, emaspeed=emaspeed)
        self.name = 'trendopen'

    def init(self, sym):
        self.sym = sym
        df = sym.df
        fast, slow = self.emaspeed[0], self.emaspeed[1]
            
        macd = Signal_MACD(df=sym.df, weight=1, fast=fast, slow=slow)
        ema = Signal_EMA(df=sym.df, weight=1, fast=fast, slow=slow)
        self.conf.addsignal([macd, ema])
        
        self.trend = Signal_Trend(df=df, offset=6, signals=[df.ema_trend], speed=self.speed)

    def entertrade(self, side, c):
        self.trade = self.inittrade(trade=Trade_TrendClose(), side=side, entryprice=c.Close, conf=self.getConfidence(side=side))
        self.trade.addCandle(c)

    def exittrade(self):
        t, c = self.trade, self.cdl
        t.marketclose.fill(c=c, price=c.Close)
        t.exittrade()

    def decide(self, c):
        self.cdl = c
        self.i = c.Index
        pxhigh, pxlow = c.pxhigh, c.pxlow

        # Exit Trade
        if not self.trade is None:
            t = self.trade
            t.addCandle(c)

            if t.side == 1:
                if c.Close < pxlow:
                    self.exittrade()                
            else:
                if c.Close > pxhigh:
                    self.exittrade()

            if not t.active:
                self.trade = None
        
        # Enter Trade
        if self.trade is None:
            pxhigh *= (1 + self.enteroffset)
            pxlow *= (1 - self.enteroffset)

            if c.Close > pxhigh:
                self.entertrade(side=1, c=c)
            elif c.Close < pxlow:
                self.entertrade(side=-1, c=c)

        # self.lasthigh, self.lastlow = pxhigh, pxlow

class Strat_Chop(Strategy):
    def __init__(self, speed=(36,36), weight=1, norm=(1,4), speedtp=(36, 36)):
        super().__init__(weight=weight)
        self.name = 'chop'
        self.speed = speed
        self.speedtp = speedtp
        self.norm = norm
        # anchordist > 0 - 0.02, step 0.002
        # Order/Stop spread, 

    def init(self, sym):
        self.sym = sym

        # unused, these moved into classes
        # sym.df = setTradePrices(self.name, sym.df, speed=self.speed)
        # sym.df = setTradePrices('tp', sym.df, speed=self.speedtp)
        # sym.df = setVolatility(sym.df, norm=self.norm)

    def decide(self, c):
        self.cdl = c

        if abs(self.status) == 1:
            self.trade.checkorders(c)
            if not self.trade.active:
                self.trade.exit()
                self.trade = None
        else:
            if c.High >= c.chop_high:
                self.status = -1
                self.entertrade(c.chop_high)
            elif c.Low <= c.chop_low:
                self.status = 1
                self.entertrade(c.chop_low)

    def inittrade(self, entryprice, side, balance=None):
        if balance is None:
            balance = self.sym.account.getBalance()

        contracts = f.getContracts(balance * self.weight, self.lev, entryprice, side, self.sym.altstatus)

        trade = Trade_Chop(c=self.cdl)
        trade.init(entryprice, contracts, self, side=side)
        return trade
    
    def entertrade(self, entryprice):
        self.trade = self.inittrade(entryprice, self.status)
        self.trade.checkorders(self.cdl)

    def getAnchorPrice(self, anchorstart, norm, side):
        return anchorstart * (1 + norm * 0.005 * side * -1)

    def getNextOrdArrays(self, anchorprice, c, side, trade=None):

        orders = OrdArray(ordtype_bot=1,
                        anchorprice=anchorprice,
                        orderspread=0.002 * c.norm,
                        trade=trade,
                        activate=True)
        
        stops = OrdArray(ordtype_bot=2,
                        anchorprice=f.getPrice(
                            -0.01 * c.norm,
                            orders.maxprice,
                            side),
                        orderspread=0.002 * c.norm,
                        trade=trade,
                        activate=False)
        
        outerprice = c.tp_high if side == 1 else c.tp_low

        takeprofits = OrdArray(ordtype_bot=3,
                        anchorprice=trade.anchorstart,
                        outerprice=outerprice,
                        orderspread=0,
                        trade=trade,
                        activate=False)

        return [orders, stops, takeprofits]

    def finalOrders(self, u, weight):
        lstOrders = []
        balance = u.totalbalancewallet * weight
        remainingcontracts = u.getPosition(self.sym.symbolbitmex)['currentQty']
        # print(remainingcontracts)

        if not self.trade is None:
            # we should be in a trade
            t = self.trade

            # rescale contracts to reflect actual user balance
            targetcontracts = f.getContracts(balance, self.lev, t.anchorstart, t.side, self.sym.altstatus)

            lstOrders.extend(t.orders.getUnfilledOrders(targetcontracts))
            lstOrders.extend(t.stops.getUnfilledOrders(targetcontracts, remainingcontracts))
            lstOrders.extend(t.takeprofits.getUnfilledOrders(targetcontracts, remainingcontracts))
            
        else:
            # not in a trade, need upper and lower order/stop arrays
            c = self.cdl

            trade_long = self.inittrade(c.chop_low, 1, balance=balance)
            lstOrders.extend(trade_long.orders.orders)
            lstOrders.extend(trade_long.stops.orders)

            trade_short = self.inittrade(c.chop_high, -1, balance=balance)
            lstOrders.extend(trade_short.orders.orders)
            lstOrders.extend(trade_short.stops.orders)

        return self.checkfinalorders(lstOrders)

    def printtrades(self, maxmin=0, maxlines=-1):
        from columnar import columnar
        data = []
        headers = ['N', 'Timestamp', 'Sts', 'Dur', 'Anchor', 'Entry', 'Exit', 'Contracts', 'Filled', 'Pnl', 'Balance']
        for i, t in enumerate(self.trades):
            if not maxmin == 0 and maxmin * t.pnlfinal <= 0: continue

            data.append([
                t.tradenum,
                '{:%Y-%m-%d %H}'.format(t.candles[0].Timestamp),
                t.status,
                t.duration(),
                '{:,.0f}'.format(t.anchorstart),
                '{:,.0f}'.format(t.entryprice),
                '{:,.0f}'.format(t.exitprice),
                '({:,} / {:,})'.format(t.filledcontracts, t.targetcontracts),
                t.allfilled(),
                '{:.2%}'.format(t.pnlfinal),
                round(t.exitbalance, 2)
            ])
        
            if i == maxlines: break

        table = columnar(data, headers, no_borders=True, justify='r', min_column_width=2)
        print(table)

class Strat_SFP(Strategy):
    def __init__(self, weight=1, lev=5):
        super().__init__(weight, lev)
        self.name = 'SFP'

    def init(self, sym=None, df=None):
        
        if not sym is None:
            self.sym = sym
            self.df = sym.df
            df = self.df
            self.a = self.sym.account
        elif not df is None:
            self.df = df

        self.minswing = 0.05
        self.stypes = dict(high=1, low=-1)

        offset = 6
        period_base = 48 #48, 96, 192

        for i in range(3):
            period = period_base * 2 ** i

            df[f'sfp_high{i}'] = df.High.rolling(period).max().shift(offset)
            df[f'sfp_low{i}'] = df.Low.rolling(period).min().shift(offset)

        ema = Signal_EMA(df=df, weight=1)
        
    def checktail(self, side, cdl):
        if cdl.tailsize(side=side) / cdl.size() > self.minswing:
            return True
        else:
            return False

    def checkswing(self, side, swingval, cdl):
        c = cdl.row
        px_max = c.High if side == 1 else c.Low

        if (side * (px_max - swingval) > 0 and 
            side * (c.Close - swingval) < 0):
            return True

    def isSwingFail(self, c=None, i=None):
        if c is None:
            if i is None:
                i = len(self.df) - 1

            c = self.df.iloc[i]

        cdl = Candle(row=c)
        self.cdl = cdl
        stypes = dict(high=1, low=-1)
        sfp = []

        # Swing High - need to check with multiple periods, largest to smallest?
        # only count further back if high/low is higher
        for k in stypes:
            side = stypes[k]
            prevmax = float('-inf') * side

            for i in range(3):
                swing = f'{k}{i}'
                # TODO: this is sketch
                if 'pandas.core.frame.Pandas' in str(type(c)):
                    swingval = c._asdict()[f'sfp_{swing}']
                else:
                    swingval = c[f'sfp_{swing}']

                if (side * (swingval - prevmax) > 0 and
                    self.checkswing(side=side, swingval=swingval, cdl=cdl) and
                    self.checktail(side=side, cdl=cdl)):
                        sfp.append(dict(name=swing, price=swingval))

                prevmax = swingval

        return sfp

    def entertrade(self, side, price, c):
        self.trade = self.inittrade(trade=Trade_SFP(), side=side, entryprice=price)
        self.trade.addCandle(c)

    def exittrade(self, price):
        self.trade.exit(price=price)
        self.trade = None

    def decide(self, c):
        stypes = self.stypes
        
        # EXIT Trade
        if not self.trade is None:
            t = self.trade
            t.addCandle(c)
            if t.duration() == 12:
                self.exittrade(price=c.Close)

        # ENTER Trade
        # if swing fail, then enter in opposite direction at close
        # if multiple swing sides, enter with candle side

        if self.trade is None:
            sfps = self.isSwingFail(c=c)
            
            if sfps:
                cdl = self.cdl
                m = {}
                for k in stypes.keys():
                    m[k] = len(list(filter(lambda x: k in x['name'], sfps)))
                
                m2 = {k:v for k,v in m.items() if v > 0}
                if len(m2) > 1:
                    swingtype = cdl.side()
                else:
                    swingtype = stypes[list(m2)[0]]

                self.entertrade(side=swingtype * -1, price=c.Close, c=c)


# TRADE
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
        self.trendchanged = False

    def init(self, price, targetcontracts, strat, conf=1, entryrow=0, side=None, temp=False):
        self.entrytarget = price
        self.entryprice = 0
        self.targetcontracts = int(targetcontracts)
        self.strat = strat
        self.sym = strat.sym
        self.conf = round(conf, 3)
        self.tradenum = strat.tradecount()
        self.timeout = strat.timeout
        self.cdl = self.sym.cdl
        self.entrybalance = self.sym.account.getBalance()
        self.i_enter = self.strat.i

        if side is None:
            self.status = strat.status
            self.side = strat.getSide() # TODO: sketch
        else:
            self.status = side
            self.side = side
        
        if not temp:
            self.strat.trades.append(self)

        self.enter()

    def exittrade(self):
        self.strat.status = 0
        self.pnlfinal = f.getPnl(self.side, self.entryprice, self.exitprice)
        self.exitbalance = self.sym.account.getBalance()
        self.i_exit = self.strat.i
        self.active = False

    def closeorder(self, price, contracts):
        
        if contracts == 0: return

        self.exitprice = (self.exitprice * self.exitcontracts + price * contracts) / (self.exitcontracts + contracts)

        if self.entryprice == 0:
            raise ValueError('entry price cant be 0!')

        self.sym.account.modify(f.getPnlXBT(contracts * -1, self.entryprice, price, self.sym.altstatus), self.cdl.Timestamp)
        
        self.exitcontracts += contracts
        self.contracts += contracts

    def closeposition(self):
        closeprice = self.sym.cdl.Open
        self.closeorder(price=closeprice, contracts=self.contracts * -1)
        self.deactivateorders(closeall=True) # this is only Trade_Chop

    def getCandle(self, i):
        return self.candles[i - 1]

    def addCandle(self, cdl):
        self.candles.append(cdl)
        self.cdl = cdl
    
    def duration(self):
        offset = -1 if self.partial else 0
        return len(self.candles) + offset

    def pnlacct(self):
        if self.exitbalance == 0:
            return 0
        else:
            return ((self.exitbalance - self.entrybalance) / self.entrybalance)

    def pnlxbt(self):
        # not used
        return f.getPnlXBT(contracts=self.filledcontracts,
                            entryprice=self.entryprice,
                            exitprice=self.exitprice,
                            isaltcoin=self.sym.altstatus)

    def pnlcurrent(self, c=None):
        if c is None: c = self.getCandle(self.duration())
        return f.getPnl(self.side, self.entryprice, c.Close)

    def pnlmaxmin(self, maxmin, firstonly=False):
        return f.getPnl(self.side, self.entryprice, self.extremum(self.side * maxmin, firstonly))

    def isgood(self):
        ans = True if self.pnlfinal > 0 else False
        return ans

    def isstopped(self):
        ans = True if self.pnlmaxmin(-1) < self.strat.stoppercent else False
        return ans

    def exitdate(self):
        return self.candles[self.duration()].timestamp
    
    def rescaleorders(self, balance):
        # need to fix 'orders' for trade_chop
        for order in self.orders:
            order.rescalecontracts(balance=balance, conf=self.conf)
    
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

    def allorders(self):
        return self.orders

    def exitorder(self):
        return list(filter(lambda x: 'close' in x.name, self.orders))[0]

    def df(self):
        return self.sym.df.iloc[self.i_enter:self.i_exit]

    def chart(self, pre=36, post=None, width=900, fast=50, slow=200):
        dur = self.duration()
        if post is None:
            post = dur if dur > 36 else 36
        f.chartorders(self.sym.df, self, pre=pre, post=post, width=width, fast=fast, slow=slow)

    def printcandles(self):
        for c in self.candles:
            print(
                c.Timestamp,
                c.Open,
                c.High,
                c.Low,
                c.Close)

    def printorders(self, orders=None):
        from columnar import columnar
        if orders is None:
            orders = self.allorders()

        data = []
        headers = ['IDX', 'Type', 'Side', 'Price', 'Cont', 'Active', 'Cancelled', 'Filled']

        for o in orders:
            ordtype_bot = o.ordarray.letter() if not o.ordarray is None else o.ordtype_bot 

            data.append([
                o.index,
                ordtype_bot,
                o.side,
                o.price,
                o.contracts,
                o.active,
                o.cancelled,
                o.filled])

        table = columnar(data, headers, no_borders=True, justify='r')
        print(table)

class Trade_TrendRev(Trade):
    def __init__(self):
        super().__init__()
        self.stopped = False
        self.enteroffset = 0
    
    def closeprice(self):
        c = self.strat.cdl
        price = c.pxhigh if self.side == 1 else c.pxlow
        # self.enteroffset = self.strat.vty.final(i=c.Index)
        self.enteroffset = c.norm_ema

        # TODO: move this to a 'slippage price' function
        return round(price * (1 + self.enteroffset * self.side), self.sym.decimalfigs)
    
    def enter(self):
        c = self.strat.cdl
        self.stoppercent = self.strat.stoppercent
        # self.enteroffset = self.strat.vty.final(i=c.Index)
        self.enteroffset = c.norm_ema

        limitbuyprice = self.entrytarget * (1 + self.enteroffset * self.side * -1)
        limitcloseprice = self.closeprice()
        self.stoppx = f.getPrice(self.stoppercent, limitbuyprice, self.side)

        contracts = int(self.targetcontracts * self.conf)

        self.limitopen = Order(
                    price=limitbuyprice,
                    side=self.side,
                    contracts=contracts,
                    activate=True,
                    ordtype_bot=1,
                    ordtype='Limit',
                    name='limitopen',
                    trade=self)

        self.stop = Order(
                    price=self.stoppx,
                    side=self.side * -1,
                    contracts=contracts,
                    activate=False,
                    ordtype_bot=2,
                    ordtype='Stop',
                    name='stop',
                    trade=self)

        self.limitclose = Order(
                    price=limitcloseprice,
                    side=self.side * -1,
                    contracts=contracts,
                    activate=False,
                    ordtype_bot=3,
                    ordtype='Limit',
                    name='limitclose',
                    execinst='Close',
                    trade=self)
    
    def checkpositionclosed(self):
        if self.limitclose.filled:
            self.active = False
    
    def checktimeout(self):
        if self.duration() >= self.timeout:
            self.timedout = True
            self.active = False
    
    def checkorders(self, c):
        # trade stays active until pxlow is hit, strat controlls
        # filling order sets the trade's actual entryprice
        # filling close or stop order sets trade's exit price        
        
        self.addCandle(c)

        for o in self.orders:
            if o.isactive(c=c) and not o.filled:
                o.check(c)

                # MARKET OPEN
                # Don't market fill if trade would have closed on this candle
                # if order not filled after 4 candles, fill at close price
                if (o.ordtype_bot == 1
                    and self.active
                    and self.duration() == 4
                    and not o.filled):
                        o.fill(c=c, price=c.Close)

                if o.filled:
                    if o.ordtype_bot == 1:
                        self.filledcontracts = o.contracts
                        delay = 1 if o.marketfilled else 0
                        self.limitclose.activate(c=c, delay=delay)
                        self.stop.activate(c=c, delay=0)
                    elif o.ordtype_bot == 2:
                        self.limitclose.cancel() #make limitclose filledtime be end of trade
                        self.stopped = True
                        self.pnlfinal = f.getPnl(self.side, self.entryprice, self.exitprice)
                        self.exitbalance = self.sym.account.getBalance()
                    elif o.ordtype_bot == 3:
                        self.stop.cancel()
            
        # adjust limitclose for next candle
        self.limitclose.setprice(price=self.closeprice())
        self.checktimeout()
        self.checkpositionclosed()

    def allorders(self):
        return [self.limitopen, self.stop, self.limitclose]

class Trade_Trend(Trade):
    def __init__(self):
        super().__init__()

    def closeprice(self):
        c = self.cdl
        price = c.pxlow if self.side == 1 else c.pxhigh

        return price

    def enter(self, temp=False):

        contracts = int(self.targetcontracts * self.conf)

        self.stopopen = Order(
                    price=self.entrytarget,
                    side=self.side,
                    contracts=contracts,
                    activate=True,
                    ordtype_bot=4,
                    ordtype='Stop',
                    name='stopopen',
                    trade=self)

        self.stopclose = Order(
                    price=self.closeprice(),
                    side=self.side * -1,
                    contracts=contracts,
                    activate=False,
                    ordtype_bot=2,
                    ordtype='Stop',
                    name='stopclose',
                    execinst='Close',
                    trade=self)

    def checkorders(self, c):
        self.addCandle(c)

        for o in self.orders:
            if o.active and not o.filled:
                o.check(c)

                if o.filled:
                    if o.ordtype_bot == 4:
                        self.filledcontracts = o.contracts
                        self.stopclose.active = True
        
        self.stopclose.price = self.closeprice()
        
class Trade_TrendClose(Trade):
    def __init__(self):
        super().__init__()

    def enter(self, temp=False):

        contracts = int(self.targetcontracts * self.conf)

        self.marketopen = Order(
                    price=self.entrytarget,
                    side=self.side,
                    contracts=contracts,
                    activate=True,
                    ordtype_bot=5,
                    ordtype='Market',
                    name='marketopen',
                    trade=self)

        self.marketclose = Order(
                    price=self.entrytarget,
                    side=self.side * -1,
                    contracts=contracts,
                    activate=False,
                    ordtype_bot=6,
                    ordtype='Market',
                    name='marketclose',
                    execinst='Close',
                    trade=self)
        
        self.marketopen.fill(c=self.cdl)
        
class Trade_Chop(Trade):
    def __init__(self, c):
        super().__init__()
        self.numorders = 4
        self.cdl = c
            
    def enter(self):
        self.anchorstart = self.entryprice
        self.entryprice = 0
        
        c = self.cdl
        
        self.anchorprice = self.strat.getAnchorPrice(self.anchorstart, c.norm, self.status)

        lst = self.strat.getNextOrdArrays(self.anchorprice, c, self.status, self)
        self.orders = lst[0]
        self.stops = lst[1]
        self.takeprofits = lst[2]

    def exit(self):
        self.filledcontracts = self.orders.filledcontracts
        self.exittrade()
        
    def checkorders(self, c):
        self.addCandle(c)

        if self.duration() == 5: # 5 is arbitrary
            self.deactivateorders()
        elif self.duration() == 40:
            self.closeposition()
        
        self.orders.checkorders(c)
        self.stops.checkorders(c)
        self.takeprofits.checkorders(c)
        
        if not self.orders.active and self.contracts == 0:
            self.active = False # > then exit trade??

        if (not self.stops.active) or (not self.takeprofits.active):
            self.active = False

    def deactivateorders(self, closeall=False):
        if not closeall:
            for i, order in enumerate(self.orders.orders):
                if not order.filled:
                    order.cancel()
                    self.takeprofits.orders[i].cancel()
                    self.stops.orders[i].cancel()
        else:
            for order in self.allorders():
                if not order.filled:
                    order.cancel()

    def allfilled(self):
        return '{}-{}-{}'.format(
            self.orders.strfilled(),
            self.stops.strfilled(),
            self.takeprofits.strfilled())

    def allorders(self):
        lst = []
        lst.extend(self.orders.orders)
        lst.extend(self.stops.orders)
        lst.extend(self.takeprofits.orders)        
        return lst

    def printallorders(self):
        self.strat.printorders(self.allorders())

class Trade_SFP(Trade):
    def __init__(self):
        super().__init__()

    def exit(self, price):
        self.marketclose.price = price # so that not 'marketfilled'
        self.marketclose.fill(c=self.cdl)
        self.exittrade()

    def enter(self):
        self.marketopen = Order(
                    price=self.entrytarget,
                    side=self.side,
                    contracts=self.targetcontracts,
                    activate=True,
                    ordtype_bot=5,
                    ordtype='Market',
                    name='marketopen',
                    trade=self)

        self.marketclose = Order(
                    price=self.entrytarget,
                    side=self.side * -1,
                    contracts=self.targetcontracts,
                    activate=True,
                    ordtype_bot=6,
                    ordtype='Market',
                    name='marketclose',
                    trade=self)

        self.marketopen.fill(c=self.cdl)


# ORDER
class Order():
    def __init__(self, price, contracts, ordtype, side=None, ordtype_bot=None, ordarray=None, trade=None, sym=None,  activate=False, index=0, symbol=None, name=None, execinst=[]):

        self.ordarray = ordarray
        self.trade = trade
        self.sym = sym

        self.symbol = symbol
        self.name = name
        self.orderID = ''
        self.ordtype = ordtype

        self.active = activate
        self.delaytime = None
        self.activenext = False # only used in ordarray, superceeded with delaytime
        self.filled = False
        self.marketfilled = False
        self.cancelled = False
        self.filledtime = None
        self.index = index # ordarray only  
        self.livedata = []

        if not isinstance(execinst, list): execinst = [execinst]
        self.execinst = []
        self.execinst.extend(execinst)
        if self.ordtype == 'Limit':
            self.execinst.append('ParticipateDoNotInitiate')

        if not self.ordarray is None:
            self.trade = self.ordarray.trade

        if not self.trade is None:
            self.sym = self.trade.strat.sym
            self.slippage = self.trade.strat.slippage
            self.trade.orders.append(self)
        
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

        # enterexit: 1,2 happen in same direction, 3,4 opposite (used to check if filled)
        # addsubtract: 1,4 add contracts to trade, 2,3 remove, relative to side
        # 1 - LimitOpen
        # 2 - StopClose
        # 3 - LimitClose
        # 4 - StopOpen
        # 5 - MarketOpen - only used for addsubtract
        # 6 - MarketClose
        if ordtype_bot:
            self.ordtype_bot = ordtype_bot
            self.enterexit = -1 if ordtype_bot in (1, 2) else 1
            self.addsubtract = -1 if ordtype_bot in (2, 3, 6) else 1
            self.isstop = True if ordtype_bot in (2, 4) else False
            self.manual = False

            if not self.trade is None:
                self.direction = self.trade.side * self.enterexit

            # TODO: this is a bit sketch
            if self.name == 'stopopen':
                self.execinst.append('IndexPrice')
            elif self.name == 'stopclose' or self.name[0] == 'S':
                self.execinst.extend(['Close', 'IndexPrice'])
            elif self.name[0] == 'T':
                self.execinst.append('Close')
        else:
            self.isstop = True if self.ordtype == 'Stop' else False
            self.manual = True
            self.name = 'manual'

        # If Side is passed explicitly > force it, else get from contracts
        if not side is None:
            self.side = side
            self.contracts = abs(contracts) * side
        else:
            self.side = 1 if contracts > 0 else -1
            self.contracts = contracts

        self.price = self.finalprice(price)
        self.pxoriginal = self.price
        self.setkey()

    def ordtype_str(self):
        # v sketch
        ans = 'L' if self.filled and not self.marketfilled else 'M'
        return ans

    def setname(self, name):
        self.name = name
        self.setkey()

    def setkey(self):
        side = self.side if self.trade is None else self.trade.side

        self.key = f.key(self.symbolbitmex, self.name, side, self.ordtype)
        self.clOrdID = '{}-{}'.format(self.key, int(time()))        

    def stoppx(self):
        return self.price * (1 + self.slippage * self.side)

    def outofrange(self, c):
        # not used
        pnl = f.getPnl(entryprice=self.price,
                        exitprice=c.Close,
                        side=self.side)

        ans = True if pnl > 0.903 else False
        return ans

    def setprice(self, price):
        if not self.filled:
            self.price = price
                                
    def checkstopprice(self):
        # use price if Limit, else use slippage price
        price = self.price if not self.isstop else self.stoppx()
        return price
                                
    def check(self, c):
        checkprice = c.High if self.direction == 1 else c.Low
        
        if self.direction * (self.price - checkprice) <= 0:
            self.fill(c=c)
            
    def open(self, price):
        t = self.trade
        contracts = self.contracts

        if contracts == 0: return

        t.entryprice = (t.entryprice * t.contracts + price * contracts) / (t.contracts + contracts)
        t.contracts += contracts            
            
    def close(self, price):
        self.trade.closeorder(price=price, contracts=self.contracts)
            
    def fill(self, c, price=None):
        self.filled = True
        self.filledtime = c.Timestamp

        # market filling
        if not price is None:
            self.price = price
            self.marketfilled = True
        
        price = self.checkstopprice()

        self.open(price=price) if self.addsubtract == 1 else self.close(price=price)
            
    def isactive(self, c):
        if not self.delaytime is None:
            active = True if self.active and c.Timestamp >= self.delaytime else False
        else:
            active = self.active

        return active        
            
    def activate(self, c, delay=0):
        self.delaytime = c.Timestamp + delta(hours=delay)
        self.active = True
            
    def printself(self):
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
        self.filledtime = self.sym.cdl.Timestamp
        if not self.ordarray is None:
            self.ordarray.openorders -= 1

    def rescalecontracts(self, balance, conf=1):
        self.contracts = int(conf * f.getContracts(
                        xbt=balance,
                        leverage=self.trade.strat.lev,
                        entryprice=self.price,
                        side=self.side,
                        isaltcoin=self.sym.altstatus))

    def intakelivedata(self, livedata):
        self.livedata = livedata
        self.orderID = livedata['orderID']
        # self.ordType = livedata['ordType']

    def append_execinst(self, m):
        if self.execinst:
            m['execInst'] = ','.join(self.execinst)
        return m

    def amendorder(self):
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
        m = self.neworder()
        m['contracts'] = self.contracts
        m['side'] = self.side
        m['name'] = self.name
        return m
    
    def neworder(self):
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
    
    def finalprice(self, price=None):
        if self.manual:
            return price
        else:
            if price is None:
                price = self.price
            return round(round(price, self.decimalfigs) + self.decimaldouble * self.side * -1, self.decimalfigs) #slightly excessive rounding
     
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
    
    def checkorders(self, c):
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

    def printorders(self):
        for order in self.orders:
            order.printself()


# OTHER
class Txn():
    def __init__(self):
        self.amount = 0
        self.timestamp = None
        self.acctbalance = 0
        self.percentchange = 0

    def printTxn(self):
        # Debug.Print Format(Me.DateTx, "yyyy-mm-dd HH"), Me.AcctBalance, Me.Amount
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
