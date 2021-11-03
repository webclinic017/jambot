# type: ignore

try:
    from datetime import datetime as dt

    import azure.functions as func
    from __app__.jambot import comm as cm
    from __app__.jambot import livetrading as live
    from __app__.jambot.database import db
    from __app__.jambot.livetrading import ExchangeManager
    from __app__.jambot.tables import Funding, Tickers
except:
    from __app__.jambot import comm as cm
    cm.send_error()
    # TODO make error wrapper or global error catcher


def main(mytimer: func.TimerRequest) -> None:
    try:
        em = ExchangeManager()

        # update 15 min candles for both bitmex + bybit
        interval = 15
        exchs = [em.default(k) for k in ('bitmex', 'bybit')]
        bmex = exchs[0]

        Tickers().update_from_exch(exchs=exchs, interval=interval)

        # Only need to update this ever 8 hrs
        d = dt.utcnow()
        is_zero_hour = 0 <= d.minute < 15

        if d.hour in (4, 12, 20) and is_zero_hour:
            Funding().update_from_exch(exchs=bmex, symbols='XBTUSD')

        live.run_strat_live(
            interval=interval,
            em=em,
            next_funding=bmex.next_funding('XBTUSD'))

        # update hourly candles after strat run on 15 min (bitmex only for now)
        if is_zero_hour:
            db.update_all_symbols(exchs=bmex, interval=1)

    except:
        cm.send_error()
