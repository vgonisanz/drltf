"""
Send a notification to your own telegram bot.

You will need to set up the following environment variables:
    TELEGRAM_NOTIFY_APIKEY: With your bot conversation
    TELEGRAM_NOTIFY_CHATID: The ID of the conversation
"""
import os
import typer

from drltf.utils import setup_logger
from drltf.utils import notify_telegram


def main(message: str):
    notify_telegram(message)

if __name__ == "__main__":
    setup_logger()
    typer.run(main)
