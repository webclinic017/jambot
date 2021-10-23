from datetime import datetime as dt  # noqa
from datetime import timedelta as delta  # noqa

import azure.functions as func
from __app__.jambot import functions as f  # type: ignore
from __app__.jambot import livetrading as live  # type: ignore # noqa
from __app__.jambot.database import db  # type: ignore
from __app__.jambot.exchanges.bitmex import Bitmex  # type: ignore
from __app__.jambot.exchanges.bybit import Bybit  # type: ignore


def main(mytimer: func.TimerRequest) -> None:
    try:
        # update 15 min candles for both bitmex + bybit
        interval = 15
        bmex = Bitmex.default(test=False, refresh=False)
        bbit = Bybit.default(test=False, refresh=False)
        exchs = [bmex, bbit]

        db.update_all_symbols(exchs=exchs, interval=interval)

        live.run_strat_live(interval=interval)

        # update hourly candles after strat run on 15 min (bitmex only for now)
        if 0 <= dt.utcnow().minute < 15:
            db.update_all_symbols(exchs=bmex, interval=1)

    except:
        f.send_error()
