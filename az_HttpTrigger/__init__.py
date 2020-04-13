import logging
import sys
from pathlib import Path

import azure.functions as func

from __app__.jambot import (
    functions as f,
    livetrading as live)

def err():
    msg = 'ERROR: Http function not triggered.'
    logging.error(msg)
    return func.HttpResponse(msg, status_code=400)

def main(req: func.HttpRequest) -> func.HttpResponse:

    try:
        action = req.params.get('action')
        logging.info(f'Python HTTP trigger function processed a request: {action}')
        if not action:
            try:
                req_body = req.get_json()
            except:
                return err()
            else:
                action = req_body.get('action')

        if action == 'refresh_balance':
            live.refresh_gsheet_balance()
        elif action == 'run_toploop':
            live.TopLoop(partial=True)
        elif action == 'close_position':
            
            pass
        elif action == 'cancel_orders':
            u = live.User()
            u.cancelmanual()
        else:
            return err()
        
        return func.HttpResponse(f'{action} success!', status_code=200)
    except:
        try:
            f.senderror()
        finally:
            return err()
        