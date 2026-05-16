# pip install dingtalk-stream
# https://opensource.dingtalk.com/developerpedia/docs/explore/tutorials/stream/bot/python/build-bot

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os

import certifi


os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from dingtalk_stream import AckMessage
import dingtalk_stream


def setup_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
def define_options():
    return argparse.Namespace(
        client_id="dingelbibniq8upo9j27",
        client_secret="DNOFcLKt8fkvm8OifpuNx3UGGwcZUuvQO1yRzy_cIYQtb7cIYPTizGBqhRByuGJf",
    )



class EchoTextHandler(dingtalk_stream.ChatbotHandler):
    def __init__(self, logger: logging.Logger = None):
        super(dingtalk_stream.ChatbotHandler, self).__init__()
        if logger:
            self.logger = logger

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        incoming_message = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
        text = incoming_message.text.content.strip()
        self.reply_text(text, incoming_message)
        return AckMessage.STATUS_OK, 'OK'

def main():
    logger = setup_logger()
    options = define_options()

    credential = dingtalk_stream.Credential(options.client_id, options.client_secret)
    client = dingtalk_stream.DingTalkStreamClient(credential)
    client.register_callback_handler(dingtalk_stream.chatbot.ChatbotMessage.TOPIC, EchoTextHandler(logger))
    client.start_forever()


if __name__ == '__main__':
    main()
