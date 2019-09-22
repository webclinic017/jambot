from datetime import datetime as date
import azure.functions as func
import sys
sys.path.append('/home/site/wwwroot')

import Functions as f
import LiveTrading as live
# test comment

def main(mytimer: func.TimerRequest) -> None:
    try:
        f.updateAllSymbols()
        live.TopLoop()
        f.discord('Updated: {}'.format(date.strftime(date.now(), f.TimeFormat(mins=True))))
    except:
        f.senderror()