
import itertools
import warnings

from bitmex import bitmex
from swagger_spec_validator.common import SwaggerValidationWarning

from jambot.exchanges.exchange import Exchange
from jambot.tradesys import orders as ords
from jambot.tradesys.__init__ import *
from jambot.tradesys.orders import BitmexOrder
from jambot.utils.secrets import SecretsManager

log = getlog(__name__)
warnings.filterwarnings('ignore', category=Warning, message='.*format is not registered')


class Bitmex(Exchange):
    div = 100000000

    def __init__(self, user: str, test: bool = False, refresh: bool = False, **kw):
        super().__init__(user=user, test=test, **kw)
        # self.name = ''
        # self.nameshort = ''
        self.percentbalance = 1
        self.avail_margin = 0
        self.total_balance_margin = 0
        self.total_balance_wallet = 0
        self.reservedbalance = 0
        self.unrealized_pnl = 0
        self.prev_pnl = 0
        self._orders = None
        self._positions = None
        self.partialcandle = None

        f.set_self(vars(), exclude=('refresh',))

        if refresh:
            self.refresh()

    @classmethod
    def default(cls, test: bool = False, **kw) -> 'Bitmex':
        """Create Bitmex obj with default name"""
        user = 'jayme' if not test else 'testnet'
        return cls(user=user, test=test, **kw)

    def load_creds(self, user: str):
        """Load creds from csv"""
        df = SecretsManager('bitmex.csv').load \
            .set_index('user')

        if not user in df.index:
            raise RuntimeError(f'User "{user}" not in saved credentials.')

        return df.loc[user].to_dict()

    def init_client(self, test: bool = False):
        warnings.simplefilter('ignore', SwaggerValidationWarning)
        return bitmex(test=test, api_key=self.key, api_secret=self.secret)

    def refresh(self):
        """Set margin balance, current position info, all orders"""
        self.set_total_balance()
        self.set_positions()
        self.set_orders()

    @property
    def orders(self):
        return self._orders

    @property
    def balance(self) -> float:
        """"""
        return self.total_balance_wallet - self.reservedbalance

    def check_request(self, request, retries: int = 0):
        """
        Perform http request and retry with backoff if failed

        - type(request) = bravado.http_future.HttpFuture
        - type(response) = bravado.response.BravadoResponse
        - response = request.response(fallback_result=[400], exceptions_to_catch=bravado.exception.HTTPBadRequest)
        """

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

    def get_position(self, symbol: str) -> dict:
        """Get position for specific symbol"""
        return self._positions.get(symbol.lower())

    def set_positions(self) -> None:
        """Set position for all symbols"""
        res = self.client.Position.Position_get().result()[0]

        # store as dict with symbol as keys
        self._positions = {}

        for pos in res:
            symbol = pos['symbol']
            self._positions[symbol.lower()] = pos

    def open_contracts(self):
        """Open contracts for each position"""
        return {k: v['currentQty'] for k, v in self._positions.items()}

    def df_orders(self, symbol=None, new_only=True, refresh=False):
        orders = self.get_orders(symbol=symbol, new_only=new_only, refresh=refresh)
        cols = ['ordType', 'name', 'size', 'price', 'execInst', 'symbol']

        if not orders:
            df = pd.DataFrame(columns=cols, index=range(1))
        else:
            df = pd.json_normalize(orders)
            df['size'] = df.orderQty * df.side
            df['price'] = np.where(df.price > 0, df.price, df.stopPx)

        df = df \
            .reindex(columns=cols) \
            .sort_values(
                by=['symbol', 'ordType', 'name'],
                ascending=[False, True, True]) \
            .reset_index(drop=True)

        return df

    def get_order_by_key(self, key):
        if self._orders is None:
            self.set_orders(new_only=True)

        # Manual orders don't have a key, not unique
        orders = list(filter(lambda x: 'key' in x.keys(), self._orders))
        orders = list(filter(lambda x: x['key'] == key, orders))

        if orders:
            return orders[0]  # assuming only one order will match given key
        else:
            return dd(type(None))

    def get_filled_orders(self, symbol='', starttime=None):

        if starttime is None:
            starttime = dt.utcnow() + delta(days=-7)

        fltr = dict(ordStatus='Filled')

        self.set_orders(fltr=fltr, new_only=False, starttime=starttime, reverse=False)
        return self._orders

    def get_orders(
            self,
            symbol: str = None,
            new_only=False,
            bot_only=False,
            manual_only=False) -> list:
        """Get orders which match criterion

        Parameters
        ----------
        symbol : str, optional
            symbol eg 'xbtusd'
        new_only : bool, optional
            unfilled orders, default False
        bot_only : bool, optional
            orders created by bot, default False
        manual_only : bool, optional
            orders not created by bot, by default False

        Returns
        -------
        list of order dicts
        """
        var = vars()

        conds = dict(
            symbol=lambda x: x['symbol'].lower() == symbol.lower(),
            new_only=lambda x: x['ordStatus'] == 'New',
            bot_only=lambda x: not x['manual'],
            manual_only=lambda x: x['manual'])

        # filter conditions based on true args in vars()
        conds = {k: v for k, v in conds.items() if not var.get(k) in (None, False)}

        return [o for o in self._orders if all(cond(o) for cond in conds.values())]

    def bitmex_order_from_raw(self, order_specs: list) -> List[BitmexOrder]:
        """Create bitmex order objs from raw specs
        - top level wrapper to both add raw specs and convert to BitmexOrder

        Parameters
        ----------
        order_specs : list
            list of order specs from bitmex

        Returns
        -------
        List[BitmexOrder]
            list of bitmex orders
        """

        return ords.make_bitmex_orders(self.add_custom_specs(order_specs))

    def add_custom_specs(self, order_specs: list) -> List[dict]:
        """Preprocess orders from exchange to add custom markers

        Parameters
        ----------
        order_specs : list
            list of order specs from bitmex

        Returns
        -------
        List[dict]
            list of order specs with custom keys added

        Raises
        ------
        RuntimeError
            if order has already been processed
        """
        if order_specs is None:
            log.warning('No orders to add into.')
            return

        for o in f.as_list(order_specs):
            if 'processed' in o.keys():
                raise RuntimeError('Order spec has already been processed!')

            o['processed'] = True
            o['sideStr'] = o['side']
            o['side'] = 1 if o['side'] == 'Buy' else -1
            o['orderQty'] = int(o['side'] * o['orderQty'])

            # add key to the order, excluding manual orders
            if not o['clOrdID'] == '':
                # o['name'] = '-'.join(o['clOrdID'].split('-')[1:-1])
                o['name'] = o['clOrdID'].split('-')[1]

                if not 'manual' in o['clOrdID']:
                    # o['key'] = '-'.join(o['clOrdID'].split('-')[:-1])
                    o['manual'] = False
                else:
                    o['manual'] = True
            else:
                o['name'] = '(manual)'
                o['manual'] = True

        return order_specs

    def set_orders(self, fltr: dict = None, count: int = 100, starttime=None, reverse=True):
        """Save last count orders from exchange"""

        if not fltr is None:
            fltr = json.dumps(fltr)

        orders = self.client.Order \
            .Order_getOrders(
                filter=fltr,
                reverse=reverse,
                count=count,
                startTime=starttime) \
            .response().result

        self._orders = self.add_custom_specs(orders)

    def funding_rate(self, symbol=SYMBOL):
        """Get current funding rate from bitmex"""
        result = self.client.Instrument.Instrument_get(symbol=symbol).response().result[0]

        rate = result['fundingRate']
        hrs = int((result['fundingTimestamp'] - f.timenow().replace(tzinfo=tz.utc)).total_seconds() / 3600)

        return rate, hrs

    def set_total_balance(self):
        """Set margin/wallet values"""
        div = self.div
        res = self.check_request(self.client.User.User_getMargin(currency='XBt'))
        self.avail_margin = res['excessMargin'] / div  # total available/unused > only used in postOrder
        self.total_balance_margin = res['marginBalance'] / div  # unrealized + realized > don't actually use
        self.total_balance_wallet = res['walletBalance'] / div  # realized
        self.unrealized_pnl = res['unrealisedPnl'] / div
        self.prev_pnl = res['prevRealisedPnl'] / div

    def _order_request(self, action: str, order_specs: list) -> List[BitmexOrder]:
        """Send order submit/amend/cancel request

        Parameters
        ----------
        action : str
            submit | amend | cancel
        order_specs : list
            list of orders to process

        Returns
        -------
        List[BitmexOrder]
            list of order results
        """
        if not order_specs:
            return

        func, param = dict(
            submit=('Order_newBulk', 'orders'),
            amend=('Order_amendBulk', 'orders'),
            cancel=('Order_cancel', 'orderID')).get(action)

        result = self.check_request(
            getattr(self.client.Order, func)(**{param: json.dumps(order_specs)}))

        resp_orders = self.bitmex_order_from_raw(result)

        # check if submit/amend orders incorrectly cancelled
        if not action == 'cancel':
            failed_orders = [o for o in resp_orders if o.is_cancelled]

            for o in failed_orders:
                msg = '***** ERROR: Order CANCELLED! \n{} \n{}' \
                    .format(o.to_json(), o.raw_spec('text'))

                f.send_error(msg, prnt=True)

        return resp_orders

    def amend_orders(self, orders: Union[list, 'BitmexOrder']) -> List[BitmexOrder]:
        """Amend order price or qty"""
        order_specs = [o.order_spec_amend for o in f.as_list(orders)]
        return self._order_request(action='amend', order_specs=order_specs)

        # except:
        #     msg = ''
        #     for order in amendorders:
        #         msg += json.dumps(order.amend_order()) + '\n'
        #     f.send_error(msg)

    def submit_simple_order(
            self,
            qty: int,
            price: float = None,
            ordtype: str = 'Limit',
            symbol: str = SYMBOL):
        """Qucik submit single order from args instead of order obj"""

        if price is None:
            ordtype = 'Market'

        order = BitmexOrder(
            price=price,
            qty=qty,
            order_type=ordtype,
            symbol=symbol)

        print('Sending new_order:')
        display(order.order_spec())

        o = self.place_bulk(placeorders=order)[0]
        display(f.useful_keys(o))

    def submit_orders(self, orders: Union[list, 'BitmexOrder']) -> List[BitmexOrder]:
        """Submit single or multiple orders

        Parameters
        ----------
        orders : list | BitmexOrder
            single or list of BitmexOrder objects

        Returns
        -------
        list
            list of orders placed
        """
        orders = f.as_list(orders)

        for o in orders:
            if not o.is_bitmex:
                raise TypeError(f'Order must be of type BitmexOrder, not {type(o)}.')

        # split orders into groups, market orders must be submitted as single
        grouped = {k: list(g) for k, g in itertools.groupby(orders, lambda x: x.is_market)}

        batches = []
        batches.append(grouped.get(False, []))

        # add each market order as it's own group
        for order in grouped.get(True, []):
            batches.append([order])

        result = []
        for order_batch in batches:
            result_orders = self._order_request(action='submit', order_specs=order_batch)
            result.extend(result_orders)

        return result

    def close_position(self, symbol: str = SYMBOL):
        try:
            # m = dict(symbol='XBTUSD', execInst='Close', ordType='Market')
            self.check_request(
                self.client.Order.Order_new(symbol=symbol, execInst='Close'))
            # self.check_request(
            #     self.client.Order.Order_newBulk(
            #         orders=json.dumps(m)))
        except:
            f.send_error(msg='ERROR: Could not close position!')

    def cancel_manual(self) -> List[BitmexOrder]:
        raise NotImplementedError('this doesn\'t work yet.')
        orders = self.get_orders(refresh=True, manual_only=True)
        self.cancel_orders(orders=orders)

    def cancel_orders(self, orders: Union[list, 'BitmexOrder']) -> List[BitmexOrder]:
        """Cancel one or multiple orders by order_id

        Parameters
        ----------
        orders : Union[list, BitmexOrder]
            list of orders to cancel

        Returns
        -------
        List[BitmexOrder]
            list of cancelled orders
        """
        orders = f.as_list(orders)
        order_specs = [o.order_id for o in orders]

        if not order_specs:
            log.warning('No orders to cancel.')
            return

        return self._order_request(action='cancel', order_specs=order_specs)

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
                length = len(df)
                cut = length - (length // 3) * 3
                if cut > 0:
                    df = df[:cut * -1]

            lst.append(df.resample('15Min', on='Timestamp').agg(
                OrderedDict([
                    ('Symbol', 'first'),
                    ('Open', 'first'),
                    ('High', 'max'),
                    ('Low', 'min'),
                    ('Close', 'last')])))

        return pd.concat(lst).reset_index()

    def get_candles(
            self,
            symbol: str = '',
            starttime=None,
            fltr='',
            retainpartial=False,
            includepartial=True,
            interval=1,
            count=1000,
            pages=100):

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
                VolBTC=lambda x: np.where(x.Symbol == 'XBTUSD', x.Homenotional, x.Foreignnotional))[cols]

        if includepartial:
            self.partialcandle = df.tail(1).copy().reset_index(drop=True)  # save last as df

            if not retainpartial:
                df.drop(df.index[-1], inplace=True)

        return df

    def printit(self, result):
        print(json.dumps(result, default=str, indent=4))