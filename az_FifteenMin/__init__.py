from datetime import datetime as dt  # noqa
from datetime import timedelta as delta  # noqa

import azure.functions as func
from __app__.jambot import SYMBOL  # type: ignore
from __app__.jambot import functions as f  # type: ignore
from __app__.jambot import livetrading as live  # type: ignore # noqa
from __app__.jambot.database import db  # type: ignore
from __app__.jambot.exchanges.bitmex import Bitmex  # type: ignore


def main(mytimer: func.TimerRequest) -> None:
    try:
        interval = 15
        exch = Bitmex.default(test=False, refresh=False)
        db.update_all_symbols(exch=exch, interval=interval, symbol=SYMBOL)

        live.run_strat_live(interval=interval)

        # update hourly candles after strat run on 15 min
        if 0 <= dt.utcnow().minute < 15:
            db.update_all_symbols(exch=exch, interval=1, symbol=SYMBOL)

    except:
        f.send_error()
