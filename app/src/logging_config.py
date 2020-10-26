import time
import functools
import datetime

import logging
from logging import getLogger, StreamHandler, Formatter
from logging.handlers import RotatingFileHandler

LOG_DIR = "../log"

is_loaded = False

LOGGER_NAMES = ["splt", "logit"]

LEVEL = logging.INFO  # logging.DEBUG

def log_it(func):
    @functools.wraps(func)
    def _log_it(*args, **kwargs):
        logger = getLogger("logit")
        extra = {"_module":func.__module__, "_funcName":func.__name__, "_lineno":func.__code__.co_firstlineno}
        logger.info("start.", extra=extra)
        st = time.time()
        res = func(*args, **kwargs)
        en = time.time()
        logger.info(f"finish. time = {en-st:.2f}(sec)", extra=extra)
        return res
    return _log_it


def init(log_file_path=None):
    global is_loaded

    if is_loaded:
        return

    if log_file_path == "auto":
        dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = f"{LOG_DIR}/log_{dt_str}.log"

    for name in LOGGER_NAMES:
        logger = getLogger(name)

        #組み込みの深刻度の中では DEBUG が一番低く、 CRITICAL が一番高くなります。
        logger.setLevel(LEVEL)

        stream_handler = StreamHandler()
        stream_handler.setLevel(LEVEL)
        if name == "logit":
            handler_format = Formatter(fmt = '[%(asctime)s][%(levelname)s][%(_module)s.%(_funcName)s %(_lineno)s] %(message)s',
                                    datefmt = "%Y/%m/%d %H:%M:%S")
        else:
            handler_format = Formatter(fmt = '[%(asctime)s][%(levelname)s][%(module)s.%(funcName)s %(lineno)s] %(message)s',
                                    datefmt = "%Y/%m/%d %H:%M:%S")
        stream_handler.setFormatter(handler_format)
        logger.addHandler(stream_handler)

        if log_file_path is not None:
            add_file_handler(logger, log_file_path, handler_format)

    is_loaded = True


def add_file_handler(logger, log_file_path, handler_format):
    file_handler = RotatingFileHandler(log_file_path, "a+", maxBytes=3*1e6, backupCount=100, encoding="utf-8")
    file_handler.setLevel(LEVEL)
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    print(f"set file handler path = {log_file_path}")


def init_pytorch():
    for name in ["pytorch_pretrained_bert.modeling", "pytorch_pretrained_bert.modeling_custom", "pytorch_pretrained_bert.tokenization"]:
        logger = getLogger(name)
        logger.setLevel(LEVEL)

        stream_handler = StreamHandler()
        stream_handler.setLevel(LEVEL)
        handler_format = Formatter('%(asctime)s - %(levelname)s - %(name)s - %(lineno)s - %(funcName)s - %(message)s')
        stream_handler.setFormatter(handler_format)
        logger.addHandler(stream_handler)
