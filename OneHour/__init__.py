import sys
from os import path
sys.path.append('/home/site/wwwroot')
sys.path.append(path.dirname(path.dirname(__file__)))

from datetime import datetime as date
from datetime import timedelta as delta

import azure.functions as func

import Functions as f
import LiveTrading as live

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
