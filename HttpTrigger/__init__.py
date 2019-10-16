import logging

import azure.functions as func
import LiveTrading as live

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    action = req.params.get('action')
    if not action:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            action = req_body.get('action')

    if action == 'refresh_balance':
        live.refresh_gsheet_balance()

        return func.HttpResponse('run success', status_code=200)
    else:
        return func.HttpResponse('Http function not triggered.', status_code=400)
