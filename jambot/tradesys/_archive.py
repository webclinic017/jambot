

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
    def __init__(self, qty=0):
        self.qty = qty
        self.orders = []

    def side(self):
        return f.side(self.qty)

    def add_order(self, orders):
        if not isinstance(orders, list):
            orders = [orders]

        for i, o in enumerate(orders):
            if isinstance(o, dict):
                o = self.convert_bitmex(o)
                orders[i] = o

            if o.ordtype == 'Market':
                self.qty += o.qty

        self.orders.extend(orders)

    def set_contracts(self, orders):
        # check bot orders that REDUCE position
        # split orders into upper and lower, loop outwards
        def traverse(orders):
            qty = self.qty
            for o in orders:
                if o.reduce_only:
                    o.qty = qty * -1

                qty += o.qty

            return orders

        upper = sorted(filter(lambda x: x.checkside == -1, orders), key=lambda x: x.price)
        lower = sorted(filter(lambda x: x.checkside == 1, orders), key=lambda x: x.price, reverse=True)

        orders = []
        orders.extend(traverse(upper))
        orders.extend(traverse(lower))

        return orders

    def final_orders(self):
        # Split orders into market and non market, process, then recombine
        ordtype = 'Market'
        orders = [o for o in self.orders if o.ordtype == ordtype]

        nonmarket = [o for o in self.orders if o.ordtype != ordtype]
        orders.extend(self.set_contracts(orders=nonmarket))

        orders = list(filter(lambda x: x.manual == False and x.qty != 0, orders))

        return orders

    def convert_bitmex(self, o):
        # Covert Bitmex manual orders to Order()
        # TODO: This doesn't preserve all attributes (eg orderID), but creates a new order from scratch.
        price = o['stopPx'] if o['ordType'] == 'Stop' else o['price']

        return Order(
            symbol=o['symbol'],
            price=price,
            qty=o['qty'],
            ordtype=o['ordType'],
            name=o['name'])


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

        price = self.anchorprice * (1 + self.orderspread * n * self.trade.status *
                                    self.enter_exit)  # TODO: this won't work, enter_exit moved to order

        return round(price, self.decimal_figs)

    def __init__(self, ordtype_bot, anchorprice, orderspread, trade, activate=False, outerprice=None):
        self.ordtype_bot = ordtype_bot
        self.anchorprice = anchorprice
        self.orderspread = orderspread
        self.trade = trade
        self.outerprice = outerprice

        if not outerprice is None:
            self.pricerange = abs(self.outerprice - self.anchorprice)
            self.orderspread = (self.pricerange / (trade.numorders + 1)) / anchorprice

        self.decimal_figs = self.trade.strat.bm.decimal_figs

        self.orders = []
        self.active = True
        self.side = self.trade.side * self.add_subtract

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
            qty = int(round(self.getFraction(i) * self.trade.targetcontracts * self.add_subtract, 0))

            order = Order(
                price=price,
                side=self.side,
                qty=qty,
                ordarray=self,
                activate=activate,
                index=i,
                symbol=self.trade.strat.bm.symbol,
                name='{}{}{}'.format(self.letter(), i + 1, modi),
                ordtype=ordtype,
                bm=self.trade.strat.bm)

            self.orders.append(order)

    def check_orders(self, c):
        if not self.active:
            return

        if self.ordtype_bot > 1 and self.trade.qty == 0:
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
                    self.filledcontracts += order.qty
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

            # rescale to qty to reflect actual user balance
            if not targetcontracts is None:
                order.qty = int(round(self.getFraction(order.index) * targetcontracts * self.add_subtract, 0))

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
                        # loop from max filled order to current order, check if we have enough qty
                        ordarray = self.trade.orders
                        maxfilled = ordarray.filledorders
                        # print('i: {}, maxfilled: {}'.format(i, maxfilled))
                        remainingcontracts = actualcontracts
                        for y in range(maxfilled - 1, i, -1):
                            # print(y, remainingcontracts, ordarray.orders[y].qty)
                            remainingcontracts -= ordarray.orders[y].qty

                        # print('remainingfinal: {}, order.contr: {}, side: {}'.format(remainingcontracts, order.qty, ordarray.side))
                        if (remainingcontracts - order.qty * -1) * ordarray.side >= 0:
                            lst.append(order)
                            # also check to fill the last order no matter what??
        return lst

    def print_orders(self):
        for order in self.orders:
            order.print_self()


class Candle():
    def __init__(self, row):
        self.row = row

    def dHC(self):
        c = self.row
        return c.high - c.close

    def dLC(self):
        c = self.row
        return c.low - c.close

    def percentOCd(self):
        c = self.row
        return (c.close - c.open) / c.open

    def percentOC(self):
        return f.percent(self.percentOCd())

    def size(self):
        c = self.row
        return abs(c.high - c.low)

    def side(self):
        c = self.row
        ans = 1 if c.close > c.open else -1
        return ans

    def tailpct(self, side):
        return self.tailsize(side=side) / self.size()

    def tailsize(self, side):
        # 1 = upper, -1 = lower
        # tailsize depends on side of candle
        c = self.row

        if side == 1:
            return c.high - max(c.close, c.open)
        elif side == -1:
            return min(c.close, c.open) - c.low

    def getmax(self, side):
        c = self.row
        if side == 1:
            return c.high
        else:
            return c.low
