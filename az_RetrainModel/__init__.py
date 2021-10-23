try:
    import azure.functions as func
    from __app__.jambot import comm as cm  # type: ignore
    from __app__.jambot.ml import storage as st  # type: ignore
except:
    from __app__.jambot import comm as cm  # type: ignore
    cm.send_error()


# NOTE might need to set this to run on deploy, packages probably not saved on new deploy
def main(mytimer: func.TimerRequest) -> None:
    try:
        st.ModelStorageManager().fit_save_models()
    except:
        cm.send_error()
