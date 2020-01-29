import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / 'Project'))

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
