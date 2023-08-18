import sys
from time import sleep
from redis.exceptions import ConnectionError
import app.shared_context as sc
from app.helper import create_index


def run():
    queue_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    queue_name = f"{sc.QUEUE_TXT}_{queue_id}"
    sc.api_logger.info(f"consuming from redis streams: {queue_name}")
    last_id_consumed = 0
    sc.api_logger.info("starting loop")
    num_embeddings = 0
    while True:
        try:
            raw_msg = sc.api_redis_cli.xread(count=1, streams={queue_name: last_id_consumed})
        except ConnectionError as exp:
            print(f"...ERROR - {type(exp)} | {exp}...")
            continue
        if not raw_msg:
            sc.api_logger.info(f"empty {queue_name} - skipping...")
            sleep(0.5)
            continue
        try:
            last_id_consumed = raw_msg[0][1][-1][0]
            msg = raw_msg[0][1][-1][1]
            decoded_data = msg.get("data".encode()).decode()
        except (UnicodeDecodeError, AttributeError, ValueError):
            sc.api_logger.info("unicode-decode error detected - skipping")
            sleep(0.5)
            continue
        if decoded_data == sc.START_TOKEN:
            sc.api_logger.info("start token detected")
            sleep(0.5)
            continue
        if decoded_data == sc.END_TOKEN:
            sc.api_logger.info("end token detected")
            sc.api_logger.info("waiting other processes finish")
            sleep(30)
            if queue_id == sc.QUEUE_MAIN:
                sc.api_logger.info("creating index on redis")
                create_index(
                    index_name=f"idx_txt",
                    distance_metric=sc.TEXT_DISTANCE_METRIC,
                    vector_field_name="embedding",
                    embedding_dimension=sc.TEXT_EMBEDDING_DIMENSION,
                    number_of_vectors=num_embeddings,
                    index_type="HNSW",
                    prefix="txt::"
                )
                sc.api_logger.info("erasing stream")
                stream_group = sc.api_redis_cli.xread(streams={queue_name: 0})
                for streams in stream_group:
                    stream_name, messages = streams
                    [sc.api_redis_cli.xdel(stream_name, i[0]) for i in messages]
            break
        key, sentence = decoded_data.split(sc.SEPARATOR)
        embeddings = sc.load_txt_model().encode(sentence[:sc.TEXT_MAX_LENGTH])
        sc.api_logger.info(f"key: {key} | embeddings shape: {embeddings.shape}")
        embeddings_bytes = embeddings.astype(sc.TEXT_EMBEDDING_TYPE).tobytes()
        # bucket = int(key) % sc.BUCKETS
        sc.api_redis_cli.hset(
            f"txt::{key}",
            mapping={
                "embedding": embeddings_bytes,
                "id": key,
                "sentence": sentence[:sc.TEXT_MAX_LENGTH]
            }
        )
        num_embeddings += 1


if __name__ == "__main__":
    sc.api_redis_cli = sc.start_queueing(manually=True)
    sc.api_logger = sc.start_encoder_logging()
    run()
