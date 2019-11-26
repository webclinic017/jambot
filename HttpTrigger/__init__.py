import logging

import azure.functions as func

import Functions as f
import LiveTrading as live


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        action = req.params.get('action')
        if not action:
            try:
                req_body = req.get_json()
            except ValueError:
                return func.HttpResponse('Http function not triggered.', status_code=400)
            else:
                action = req_body.get('action')

        if action == 'refresh_balance':
            live.refresh_gsheet_balance()
            return func.HttpResponse('refresh balance success', status_code=200)
        elif action == 'run_toploop':
            live.TopLoop(partial=True)
            return func.HttpResponse('run TopLoop success', status_code=200)
        else:
            return func.HttpResponse('Http function not triggered.', status_code=400)
    except:
        f.senderror()
