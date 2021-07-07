"""
Run at 12pm every day to retrain model
"""
from datetime import datetime as dt
from datetime import timedelta as delta

import azure.functions as func
from __app__.jambot import functions as f  # type: ignore
from __app__.jambot import sklearn_utils as sk  # type: ignore
# from __app__.jambot import livetrading as live
from __app__.jambot.database import db  # type: ignore


def main(mytimer: func.TimerRequest) -> None:
    try:
        pass
        # u = live.User()
        # db.update_all_symbols(u=u, interval=1)

        # startdate, daterange = dt.now().date() + delta(days=-15), 30
        # dfall = db.get_dataframe(symbol='XBTUSD', startdate=startdate, daterange=daterange)

        # live.run_toploop(u=u, dfall=dfall)
        # live.check_sfp(df=dfall)
    except:
        f.send_error()
