try:
    import azure.functions as func
    from __app__.jambot import comm as cm  # type: ignore
    from __app__.jambot import livetrading as live  # type: ignore
except:
    from __app__.jambot import comm as cm  # type: ignore
    cm.send_error()


def main(mytimer: func.TimerRequest) -> None:
    try:
        live.check_filled_orders()
    except:
        cm.send_error()
