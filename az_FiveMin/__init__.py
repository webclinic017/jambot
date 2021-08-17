import azure.functions as func
from __app__.jambot import functions as f  # type: ignore

# from __app__.jambot import livetrading as live  # type: ignore


def main(mytimer: func.TimerRequest) -> None:
    try:
        pass
        # live.check_filled_orders(refresh=True)
    except:
        f.send_error()
