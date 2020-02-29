import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / 'Project'))

import azure.functions as func

import functions as f
import livetrading as live


def main(mytimer: func.TimerRequest) -> None:
    try:
        live.checkfilledorders(refresh=True)
    except:
        f.senderror()
