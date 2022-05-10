"""
Logs and time utils
"""
import os
from functools import wraps
from time import time

from telegram.ext import Updater

import logging
import structlog


logger = structlog.get_logger(__file__)


def setup_logger():
    logging.root.handlers = []
    logging.basicConfig(format='%(asctime)s - %(name)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)

    logging.getLogger("pyvirtualdisplay").setLevel(logging.INFO)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info("timeit", name=f.__name__, t_in_s=round(te-ts, 2))
        return result
    return wrap


def notify_telegram(message: str):
    """
    TELEGRAM_NOTIFY_APIKEY: Use telegram @BotFather to create a new one. And start a conversation.
    TELEGRAM_NOTIFY_CHATID: Get the conversation with some dirty hack like
        use: https://api.telegram.org/bot<TELEGRAM_NOTIFY_APIKEY>/getUpdates
    """
    apikey = os.environ.get('TELEGRAM_NOTIFY_APIKEY')
    chatid = os.environ.get('TELEGRAM_NOTIFY_CHATID')

    updater = Updater(token=apikey, use_context=True)
    updater.bot.sendMessage(chat_id=chatid, text=message)
