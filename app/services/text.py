import sys
from time import sleep
from pathlib import Path
import app.shared_context as sc
from app.helper import create_flat_index
from sentence_transformers import SentenceTransformer


def run():
    queue_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    queue_name = f"{sc.QUEUE_TXT}_{queue_id}"
    sc.api_logger.info("downloading tranformer model")
    model = SentenceTransformer(
        model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        cache_folder=str(Path(__file__).parent.parent / "dl_models"),
    )
    sc.api_logger.info(f"subscribing to redis queue: {queue_name}")
    p_txt = sc.api_redis_cli.pubsub()
    p_txt.subscribe(queue_name)

    sc.api_logger.info("starting loop")
    num_embeddings, num_empty_loops = 0, 0
    while True:
        msg = p_txt.get_message()
        if not msg:
            sc.api_logger.info(f"empty {queue_name} - skipping #{num_empty_loops}")
            if num_embeddings > 0:
                num_empty_loops += 1
                if num_empty_loops >= sc.MAX_LOOPS_WITHOUT_DATA:
                    if queue_id == 0:  # only queue_id = 0 will be responsible to create index
                        sc.api_logger.info("creating index on redis")
                        create_flat_index(sc.TEXT_EMBEDDING_FIELD_NAME, num_embeddings)
                    num_embeddings, num_empty_loops = 0, 0
            sleep(0.5)
            continue
        try:
            key, sentence = msg.get("data").decode().split(sc.SEPARATOR)
        except (UnicodeDecodeError, AttributeError, ValueError):
            sc.api_logger.info(f"unicode-decode error detected - skipping")
            sleep(0.5)
            continue
        embeddings = model.encode(sentence[:sc.TEXT_MAX_LENGTH])
        sc.api_logger.info(f"key: {key} | embeddings shape: {embeddings.shape}")
        embeddings_bytes = embeddings.astype(sc.TEXT_EMBEDDING_TYPE).tobytes()
        sc.api_redis_cli.hset(
            f"txt-{key}", mapping={"embedding": embeddings_bytes, "id": key, "sentence": sentence[:sc.TEXT_MAX_LENGTH]}
        )
        num_embeddings += 1


if __name__ == "__main__":
    sc.api_redis_cli = sc.start_queueing()
    sc.api_logger = sc.start_encoder_logging()
    run()
