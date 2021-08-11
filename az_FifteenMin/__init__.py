
from datetime import datetime as dt
from datetime import timedelta as delta

import azure.functions as func
from __app__.jambot import SYMBOL  # type: ignore
from __app__.jambot import functions as f  # type: ignore
from __app__.jambot import livetrading as live  # type: ignore
from __app__.jambot.database import db  # type: ignore
from __app__.jambot.exchanges.bitmex import Bitmex  # type: ignore


def main(mytimer: func.TimerRequest) -> None:
    try:
        exch = Bitmex.default(test=False)
        db.update_all_symbols(exch=exch, interval=15, symbol=SYMBOL)

        startdate, daterange = dt.now().date() + delta(days=-15), 30
        dfall = db.get_dataframe(symbol='XBTUSD', startdate=startdate, daterange=daterange)

        # live.run_toploop(u=u, dfall=dfall)
        # live.check_sfp(df=dfall)
    except:
        f.send_error()
