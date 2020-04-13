
from datetime import (datetime as date, timedelta as delta)

import azure.functions as func

from __app__.jambot import (
    functions as f,
    livetrading as live)

def main(mytimer: func.TimerRequest) -> None:
    try:
        u = live.User()
        f.updateAllSymbols(u=u)

        startdate, daterange = date.now().date() + delta(days=-15), 30
        dfall = f.getDataFrame(symbol='XBTUSD', startdate=startdate, daterange=daterange)

        live.TopLoop(u=u, dfall=dfall)
        live.checksfp(df=dfall)
    except:
        f.senderror()
