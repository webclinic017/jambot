from datetime import datetime as date
import azure.functions as func
import sys
sys.path.append('/home/site/wwwroot')

import Functions as f
import LiveTrading as live

def main(mytimer: func.TimerRequest) -> None:
    try:
        live.checkfilledorders()
        # f.discord('test from local host: {}'.format(date.strftime(date.now(), f.TimeFormat())))
    except:
        f.senderror()