import sys
from os import path
sys.path.append('/home/site/wwwroot')
sys.path.append(path.dirname(path.dirname(__file__)))

from datetime import datetime as date

import azure.functions as func

import Functions as f
import LiveTrading as live


def main(mytimer: func.TimerRequest) -> None:
    try:
        live.checkfilledorders(refresh=True)
    except:
        f.senderror()
