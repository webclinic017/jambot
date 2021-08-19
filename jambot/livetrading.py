from jambot import *
from jambot import config as cf
from jambot import functions as f
from jambot.exchanges.bitmex import Bitmex
from jambot.ml import models as md
from jambot.ml.storage import ModelStorageManager
from jambot.tradesys import backtest as bt
from jambot.tradesys.strategies import ml, sfp, trendrev


def compare_state(strat, pos):
    # return TRUE if side is GOOD
    # Could also check current qty?
    # only works for trend, don't use for now
    qty = pos['currentQty']
    side = f.side(qty)

    ans = True if side == 0 or strat.get_side() == side else False

    if not ans:
        err = '{}: {}, expected: {}'.format(strat.bm.symbolshort, side, strat.get_side())
        f.discord(err)

    return ans


def refresh_gsheet_balance(u=None):
    sht = f.get_google_sheet()
    ws = sht.worksheet_by_title('Bitmex')
    df = ws.get_as_df(start='A1', end='J15')
    lst = list(df['Sym'].dropna())
    syms = []

    p = cf.p_res / 'symbols.csv'
    df2 = pd.read_csv(p)
    for row in df2.itertuples():
        if row.symbolshort in lst:
            syms.append(bt.BacktestManager(symbol=row.symbol, row=row))

    # if u is None:
    #     u = User()
    # write_balance_google(syms, u, sht=sht, ws=ws, df=df)


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

    if msg:
        f.discord(msg=msg, channel='sfp')


def check_filled_orders(minutes=5, refresh=True, u=None):
    if u is None:
        # u = User()
        u = None
    starttime = dt.utcnow() + delta(minutes=minutes * -1)
    orders = u.get_filled_orders(starttime=starttime)

    if orders:
        p = cf.p_res / 'symbols.csv'
        df = pd.read_csv(p)

        lst, syms, templist = [], [], []
        nonmarket = False

        for o in orders:
            symbol, name = o['symbol'], o['name']
            row = df[df['symbolbitmex'] == symbol]
            figs = 0
            if len(row) > 0:
                symshort = row['symbolshort'].values[0]
                figs = row['decimal_figs'].values[0]

            price = o['price']
            avgpx = round(o['avgPx'], figs)

            # check for non-market buys
            if not o['ordType'] == 'Market':
                nonmarket = True

                # need to have all correct symbols in symbols.csv
                if not symbol in templist:
                    templist.append(symbol)
                    syms.append(bt.BacktestManager(symbol=symbol))

            ordprice = f' ({price})' if not price == avgpx else ''
            stats = f' | Bal: {u.total_balance_margin:.3f} | PnL: {u.prev_pnl:.3f}' if any(
                s in name for s in ('close', 'stop')) else ''

            lst.append('{} | {} {:,} at ${:,}{} | {}{}'.format(
                symshort,
                o['sideStr'],
                o['qty'],
                avgpx,
                ordprice,
                name,
                stats))

        # write balance to google sheet, EXCEPT on market buys
        # NOTE probably don't need to do this now
        if nonmarket and refresh:
            run_toploop(u=u, partial=True)
            # write_balance_google(syms, u, preservedf=True)

        msg = '\n'.join(lst)
        f.discord(msg=msg + '\n@here', channel='orders')
        # return msg


def write_balance_google(syms, u, sht=None, ws=None, preservedf=False, df=None):

    if sht is None:
        sht = f.get_google_sheet()
    if ws is None:
        ws = sht.worksheet_by_title('Bitmex')

    if df is None:
        if not preservedf:
            df = pd.DataFrame(columns=['Sym', 'Size', 'Entry', 'Last', 'Pnl', '%',
                              'ROE', 'Value', 'Dur', 'Conf'], index=range(14))
        else:
            df = ws.get_as_df(start='A1', end='J15')

    u.set_positions()

    for i, bm in enumerate(syms):
        symbol = bm.symbolbitmex
        figs = bm.decimal_figs
        pos = u.get_position(symbol)

        df.at[i, 'Sym'] = bm.symbolshort
        if bm.tradingenabled:
            df.at[i, 'Size'] = pos['currentQty']
            df.at[i, 'Entry'] = round(pos['avgEntryPrice'], figs)
            df.at[i, 'Last'] = round(pos['lastPrice'], figs)
            df.at[i, 'Pnl'] = round(pos['unrealisedPnl'] / u.div, 3)
            df.at[i, '%'] = f.percent(pos['unrealisedPnlPcnt'])
            df.at[i, 'ROE'] = f.percent(pos['unrealisedRoePcnt'])
            df.at[i, 'Value'] = pos['maintMargin'] / u.div

        if bm.strats:
            strat = bm.strats[0]
            t = strat.trades[-1]
            df.at[i, 'Dur'] = t.duration()
            df.Conf[i] = t.conf

    # set profit/balance
    u.set_total_balance()
    df.at[9, 'Size'] = u.unrealized_pnl
    df.at[9, 'Entry'] = u.total_balance_margin

    # set funding rate
    rate, hrs = u.funding_rate()
    df.at[12, 'Sym'] = 'Funding:'
    df.at[12, 'Size'] = f.percent(rate)
    df.at[12, 'Entry'] = hrs

    # set current time
    df.at[13, 'Sym'] = 'Last:'
    df.at[13, 'Size'] = dt.strftime(dt.utcnow(), f.time_format(mins=True))

    # concat last 10 trades for google sheet
    bm = list(filter(lambda x: x.symbol == 'XBTUSD', syms))[0]
    if bm.strats:
        dfTrades = bm.strat.df_trades(last=10).drop(columns=['N', 'Contracts', 'Bal'])
        dfTrades.timestamp = dfTrades.timestamp.dt.strftime('%Y-%m-%d %H')
        dfTrades.Pnl = dfTrades.Pnl.apply(lambda x: f.percent(x))
        dfTrades.PnlAcct = dfTrades.PnlAcct.apply(lambda x: f.percent(x))
    else:
        dfTrades = ws.get_as_df(start='Q1', end='Y14')  # df.loc[:, 'timestamp':'PnlAcct']

    df = pd.concat([df, u.df_orders(refresh=True), dfTrades], axis=1)
    ws.set_dataframe(df, (1, 1), nan='')
    # return df


def run_toploop(u=None, partial=False, dfall=None):
    # run every 1 hour, or when called by check_filled_orders()
    from jambot.database import db

    # Google - get user/position info
    sht = f.get_google_sheet()
    g_usersettings = sht.worksheet_by_title('UserSettings').get_all_records()
    p = cf.p_res / 'symbols.csv'
    dfsym = pd.read_csv(p)
    g_user = g_usersettings[0]
    syms = []

    # Bitmex - get user/position info
    if u is None:
        # u = User()
        u = None

    u.set_positions()
    u.set_orders()
    u.reserved_balance = g_user['Reserved Balance']  # could just pass g_user to User()

    # TODO: filter dfall to only symbols needed, don't pull everything from db
    # use 'WHERE symbol in []', try pypika
    # Only using XBTUSD currently
    startdate = f.timenow() + delta(days=-15)
    if dfall is None:
        dfall = db.get_df(symbol=SYMBOL, startdate=startdate, interval=1)

    for row in dfsym.itertuples():
        if not row.symbol == 'XBTUSD':
            continue
        try:
            # match google user with bitmex position, add %balance
            weight = float(g_user[row.symbolshort].strip('%')) / 100
            pos = u.get_position(row.symbolbitmex)
            pos['percentbalance'] = weight

            symbol = row.symbol
            # NOTE may need to set timestamp/symbol index when using more than just XBTUSD
            df = dfall[dfall.symbol == symbol]  # .reset_index(drop=True)

            # TREND_REV
            speed = (16, 6)
            norm = (0.004, 0.024)
            strat = trendrev.Strategy(speed=speed, norm=norm)
            strat.stop_pct = -0.03
            strats = [strat]

            bm = bt.BacktestManager(symbol=symbol, startdate=startdate, strats=strats,
                                    row=row, df=df, partial=partial, u=u)
            if weight <= 0:
                bm.tradingenabled = False  # this should come from strat somehow
            bm.decide_full()
            syms.append(bm)

            if bm.tradingenabled:
                actual = u.get_orders(bm.symbolbitmex, bot_only=True)
                theo = strat.final_orders(u, weight)

                # matched, missing, not_matched = compare_orders(theo, actual, show=True)

                # u.cancel_bulk(not_matched)
                # u.amend_bulk(validate_matched(matched, show=True))
                # u.place_bulk(missing)

        except:
            f.send_error(symbol)

    write_balance_google(syms, u, sht)


def run_strat_live(exch: Bitmex = None, test: bool = False, interval: int = 15) -> None:
    """Run strategy on given interval and adjust orders

    Parameters
    ----------
    exch : Bitmex, optional
        exch obj, default None
    test : bool, optional
        use testnet, by default False
    """
    # TODO add errlog wrapper
    from jambot.database import db

    if exch is None:
        exch = Bitmex.default(test=test, refresh=True)

    # trigger every 15min at 15sec
    symbol = SYMBOL
    name = 'lgbm'

    # load raw data from db
    # add signals to raw data
    offset = {1: 16, 15: 4}.get(interval)
    startdate = f.timenow(interval) + delta(days=-offset)
    df = db.get_df(symbol=symbol, startdate=startdate, interval=interval) \
        .pipe(md.add_signals, name=name)

    # load saved/trained models from blob storage and add pred signals
    df_pred = ModelStorageManager() \
        .df_pred_from_models(df=df, name=name)

    # run strat in "live" mode to get expected state
    strat = ml.make_strat(live=True)

    cols = ['open', 'high', 'low', 'close', 'signal']
    bm = bt.BacktestManager(
        symbol=symbol,
        startdate=df.index[0],
        strat=strat,
        df=df_pred[cols])

    bm.run()

    # reconcile orders
    exch.reconcile_orders(
        symbol=symbol,
        expected_orders=strat.broker.expected_orders(exch=exch, symbol=symbol))
