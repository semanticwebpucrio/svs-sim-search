import numpy as np
from time import sleep
from pathlib import Path
import app.shared_context as sc
from redis.commands.search.field import VectorField, TextField
from sentence_transformers import SentenceTransformer


def create_flat_index(vector_field_name, number_of_vectors):
    sc.api_redis_cli.ft().create_index([
        VectorField(
            vector_field_name,
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": sc.TEXT_EMBEDDING_DIMENSION,
                "DISTANCE_METRIC": sc.TEXT_DISTANCE_METRIC,
                "INITIAL_CAP": number_of_vectors,
                "BLOCK_SIZE": number_of_vectors,
            }
        ),
        TextField("id"),
        TextField("sentence"),
    ])


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
    num_embeddings, num_empty_loops = 0, 0
    while True:
        msg = p_txt.get_message()
        if not msg:
            sc.api_logger.info(f"empty {sc.QUEUE_TXT} - skipping #{num_empty_loops}")
            if num_embeddings > 0:
                num_empty_loops += 1
                if num_empty_loops >= sc.MAX_LOOPS_WITHOUT_DATA:
                    sc.api_logger.info(f"creating index on redis")
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
        embeddings_bytes = embeddings.astype(np.float64).tobytes()
        sc.api_redis_cli.hset(
            f"txt-{key}", mapping={"embedding": embeddings_bytes, "id": key, "sentence": sentence[:sc.TEXT_MAX_LENGTH]}
        )
        num_embeddings += 1


if __name__ == "__main__":
    sc.api_redis_cli = sc.start_queueing()
    sc.api_logger = sc.start_encoder_logging()
    run()
