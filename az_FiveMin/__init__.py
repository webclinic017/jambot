import azure.functions as func

from __app__.jambot import (
    functions as f,
    livetrading as live)


def main(mytimer: func.TimerRequest) -> None:
    try:
        live.checkfilledorders(refresh=True)
    except:
        f.senderror()
