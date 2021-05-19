import logging

import azure.functions as func
from __app__.jambot import functions as f
from __app__.jambot import livetrading as live


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
            live.run_toploop(partial=True)
        elif action == 'close_position':

            pass
        elif action == 'cancel_orders':
            u = live.User()
            u.cancel_manual()
        else:
            return err()

        return func.HttpResponse(f'{action} success!', status_code=200)
    except:
        try:
            f.send_error()
        finally:
            return err()
