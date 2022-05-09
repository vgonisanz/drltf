"""
Logs and time utils
"""
from functools import wraps
from time import time
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
