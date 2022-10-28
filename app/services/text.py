import numpy as np
from time import sleep
from pathlib import Path
import app.shared_context as sc
from sentence_transformers import SentenceTransformer


def run():
    sc.api_logger.info("downloading tranformer model")
    model = SentenceTransformer(
        model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        cache_folder=str(Path(__file__).parent.parent / "dl_models"),
    )
    sc.api_logger.info("subscribing to redis queue")
    p_txt = sc.api_redis_cli.pubsub()
    p_txt.subscribe(sc.QUEUE_TXT)

    sc.api_logger.info("starting loop")
    while True:
        msg = p_txt.get_message()
        if not msg:
            sc.api_logger.info(f"empty {sc.QUEUE_TXT} - skipping")
            sleep(0.5)
            continue
        try:
            key, sentence = msg.get("data").decode().split(sc.SEPARATOR)
        except (UnicodeDecodeError, AttributeError, ValueError):
            sc.api_logger.info(f"unicode-decode error detected - skipping")
            sleep(0.5)
            continue
        embeddings = model.encode(sentence)
        sc.api_logger.info(f"key: {key} | embeddings shape: {embeddings.shape}")
        embeddings_str = np.array2string(embeddings, precision=10, separator=',', suppress_small=True)
        sc.api_redis_cli.set(f"txt-{key}", embeddings_str)


if __name__ == "__main__":
    sc.api_redis_cli = sc.start_queueing()
    sc.api_logger = sc.start_encoder_logging()
    run()
