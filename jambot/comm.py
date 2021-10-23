import logging
import re
import traceback
from datetime import datetime as dt
from typing import *

from jambot.config import AZURE_WEB

try:
    from jambot import getlog
    log = getlog(__name__)
except Exception as e:
    log = logging.Logger(__name__)


def discord(msg: str, channel: str = 'jambot', log: Callable = None) -> None:
    """Send message to discord channel

    Parameters
    ----------
    msg : str
    channel : str, optional
        discord channel, default 'jambot'
    log : Callable, logging.Logger.log_func
        log the message as well, default None
    """
    from discord import RequestsWebhookAdapter, Webhook

    from jambot.utils.secrets import SecretsManager

    if not log is None:
        log(msg)

    r = SecretsManager('discord.csv').load.set_index('channel').loc[channel]
    if channel == 'err':
        msg = f'{msg} @here'

    # Create webhook
    webhook = Webhook.partial(r.id, r.token, adapter=RequestsWebhookAdapter())

    # check for py codeblocks
    expr = r'(\`\`\`py.+?\`\`\`)'
    expr2 = r'\`\`\`py\n(.+?)\`\`\`'
    res = re.split(expr, msg, flags=re.MULTILINE | re.DOTALL)

    # split_n = lambda msg, n: [msg[i:i + n] for i in range(0, len(msg), n)]

    def split_n(msg, n):
        """split msg into groups less than length n, considering newlines"""
        temp = ''
        out = []

        for msg in msg.split('\n'):
            if len(temp) + len(msg) > n:
                out.append(temp)
                temp = ''

            temp += f'\n{msg}'

        return out + [temp]

    # split into strings of max 2000 char for discord
    out = []
    for msg in res:
        if '```py' in msg:
            # extract long error
            msg = re.search(expr2, msg, flags=re.MULTILINE | re.DOTALL).groups()[0]
            split = [py_codeblock(s) for s in split_n(msg, 1980)]
        else:
            split = split_n(msg, 2000)

        out.extend(split)

    temp = ''
    final = []
    for msg in out:
        if len(temp) + len(msg) > 2000:
            final.append(temp)
            temp = ''

        temp += msg

    for msg in final + [temp]:
        webhook.send(msg)


def py_codeblock(msg: str) -> str:
    """Wrap message in a ```py codeblock for discord messaging

    Parameters
    ----------
    msg : str

    Returns
    -------
    str
        msg with py codeblock wrapper
    """
    return f'```py\n{msg}```'


def send_error(
        msg: str = None,
        prnt: bool = False,
        force: bool = False,
        _log: logging.Logger = None) -> None:
    """Send error message to discord with last err traceback

    Parameters
    ----------
    msg : str, optional
        message header info, by default None
    prnt : bool, optional
        print to stdout or send to discord, by default False
    force : bool, optional
        always send to discord, by default False
    """
    err = traceback.format_exc().replace('Traceback (most recent call last):\n', '')

    to_discord = True if AZURE_WEB or force else False

    # wrap err traceback in py code block formatting
    if to_discord:
        err = f'{py_codeblock(err)}{dt.utcnow():%Y-%m-%d %H:%M:%S}'

    # add custom msg to traceback if paassed
    msg = err if msg is None else f'{msg}\n{err}'

    # print if local dev, else send to discord
    if prnt or not to_discord:
        try:
            _logger = log if _log is None else _log
            _logger.error(msg)
        except Exception as e:
            print(f'Couldn\'t log error: {msg}')
    else:
        discord(msg=msg, channel='err')
