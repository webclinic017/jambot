# type: ignore

try:
    import azure.functions as func
    from __app__.jambot import comm as cm
    from __app__.jambot.livetrading import ExchangeManager
    from __app__.jambot.ml import storage as st
except:
    from __app__.jambot import comm as cm
    cm.send_error()


# NOTE might need to set this to run on deploy, packages probably not saved on new deploy
def main(mytimer: func.TimerRequest) -> None:
    try:
        em = ExchangeManager()
        st.ModelStorageManager().fit_save_models(em=em)
    except:
        cm.send_error()
