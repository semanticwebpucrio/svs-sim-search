import sys
import redis
import torch
import logging
import uvicorn
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi.logger import logger
from torchvision import transforms
from sentence_transformers import SentenceTransformer


# TODO: load values from .env file
# constants
REDIS_HOST = "redis"
REDIS_PORT = 6379

QUEUE_TXT = "txt_queue"
QUEUE_IMG = "img_queue"
QUEUE_MAIN = 0  # queue_id responsible to create index
QUEUES_AVAILABLE = 1
SEPARATOR = "|###|"
BUCKETS = 5

MAX_LOOPS_WITHOUT_DATA = 120  # approximate 1min
TEXT_MAX_LENGTH = 512
TEXT_EMBEDDING_DIMENSION = 768
TEXT_EMBEDDING_FIELD_NAME = "embedding"
TEXT_EMBEDDING_TYPE = np.float32
TEXT_DISTANCE_METRIC = "L2"
TEXT_INDEX_NAME = "idx_txt"

IMG_EMBEDDING_DIMENSION = 1000
IMG_EMBEDDING_FIELD_NAME = "embedding"
IMG_EMBEDDING_TYPE = np.float32
IMG_DISTANCE_METRIC = "L2"
IMG_INDEX_NAME = "idx_img"

API_PORT = 8080
API_HOST = "0.0.0.0"
API_DESCRIPTION = """
## Multimodal Clustering using Product Quantization 🚀
"""

# single instances
api_app = None
api_logger = None
api_redis_cli = None
model_txt = None
model_img = None


def start_queueing(manually=False):
    redis_client = redis.Redis(
        host=REDIS_HOST if not manually else 'localhost',
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


def encode_image(img_path: Path = None, input_image: Image = None):
    global model_img
    # lazy loading
    if not model_img:
        model_img = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", weights=True
        )
        model_img.eval()
    if not input_image:
        input_image = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model_img(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    embeddings = torch.nn.functional.softmax(output[0], dim=0)
    return embeddings


def load_txt_model():
    global model_txt
    # lazy loading
    if not model_txt:
        model_txt = SentenceTransformer(
            model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            cache_folder=str(Path(__file__).parent / "dl_models"),
        )
    return model_txt
