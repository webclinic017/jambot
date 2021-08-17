import azure.functions as func
from __app__.jambot import functions as f  # type: ignore
from __app__.jambot.ml import storage as st  # type: ignore


# NOTE might need to set this to run on deploy, packages probably not saved on new deploy
def main(mytimer: func.TimerRequest) -> None:
    try:
        st.ModelStorageManager().fit_save_models()
    except:
        f.send_error()
