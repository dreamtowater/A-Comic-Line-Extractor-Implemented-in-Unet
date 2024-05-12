import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(name, log_path):
    """
    name: logger名,建议以device命名,如'cpu'、'cuda:0'。
    log_path: 日志文件路径,文件名也建议以device命名,但要替换非法字符。

    注意:
    在ipynb文件中每次调用该函数前需重启内核,否则会重复输出。
    (因为logging.getLogger只有在内存中找不到logger时才新建logger,addHandler会重复执行)
    若无logger.addHandler(console_handler)则不在控制台输出,多卡时可设置成只有主卡才在控制台输出。
    """

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    with open(log_path, 'a'): pass  # 验证文件名合法性
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
