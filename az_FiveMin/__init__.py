import azure.functions as func
from __app__.jambot import functions as f  # type: ignore
from __app__.jambot import livetrading as live  # type: ignore

# from __app__.jambot.exchanges.bitmex import Bitmex  # type: ignore


def main(mytimer: func.TimerRequest) -> None:
    try:
        # exch = Bitmex.default(test=True, refresh=False)
        live.check_filled_orders()
    except:
        f.send_error()
