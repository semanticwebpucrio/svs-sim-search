import sys
import redis
import logging
import uvicorn
import numpy as np
from fastapi.logger import logger


# TODO: load values from .env file
# constants
REDIS_HOST = "redis"
REDIS_PORT = 6379

QUEUE_TXT = "txt_queue"
QUEUE_IMG = "img_queue"
QUEUES_AVAILABLE = 1
SEPARATOR = "|###|"

MAX_LOOPS_WITHOUT_DATA = 120  # approximate 1min
TEXT_MAX_LENGTH = 512
TEXT_EMBEDDING_DIMENSION = 768
TEXT_EMBEDDING_FIELD_NAME = "embedding"
TEXT_EMBEDDING_TYPE = np.float32
TEXT_DISTANCE_METRIC = "COSINE"

API_PORT = 8080
API_HOST = "0.0.0.0"
API_DESCRIPTION = """
## Multimodal Clustering using Product Quantization 🚀
"""

# single instances
api_app = None
api_logger = None
api_redis_cli = None


def start_queueing():
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
    )
    return redis_client


def start_api_logging():
    uvicorn_logger = logging.getLogger("uvicorn.access")
    logger.handlers = uvicorn_logger.handlers
    console_formatter = uvicorn.logging.ColourizedFormatter("{message}", style="{", use_colors=False)
    logger.handlers[0].setFormatter(console_formatter)
    logger.setLevel(uvicorn_logger.level)
    return logger


def start_encoder_logging():
    encoder_logger = logging.getLogger()
    encoder_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:\t%(message)s')
    handler.setFormatter(formatter)
    encoder_logger.addHandler(handler)
    return encoder_logger
