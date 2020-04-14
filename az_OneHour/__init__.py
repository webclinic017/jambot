
from datetime import (datetime as dt, timedelta as delta)

import azure.functions as func

from __app__.jambot import (
    functions as f,
    livetrading as live)
from __app__.jambot.database import db

def main(mytimer: func.TimerRequest) -> None:
    try:
        u = live.User()
        db.update_all_symbols(u=u)

        startdate, daterange = dt.now().dt() + delta(days=-15), 30
        dfall = db.get_dataframe(symbol='XBTUSD', startdate=startdate, daterange=daterange)

        live.run_toploop(u=u, dfall=dfall)
        live.check_sfp(df=dfall)
    except:
        f.send_error()
