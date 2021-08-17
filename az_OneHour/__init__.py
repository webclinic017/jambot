from datetime import datetime as dt  # noqa
from datetime import timedelta as delta  # noqa

import azure.functions as func
from __app__.jambot import functions as f  # type: ignore  # noqa
from __app__.jambot import livetrading as live  # type: ignore  # noqa
from __app__.jambot.database import db  # type: ignore  # noqa


def main(mytimer: func.TimerRequest) -> None:
    pass
    # try:
    #     u = live.User()
    #     db.update_all_symbols(u=u, interval=1)

    #     startdate, daterange = dt.now().date() + delta(days=-15), 30
    #     dfall = db.get_df(symbol='XBTUSD', startdate=startdate, daterange=daterange)

    #     live.run_toploop(u=u, dfall=dfall)
    #     live.check_sfp(df=dfall)
    # except:
    #     f.send_error()
