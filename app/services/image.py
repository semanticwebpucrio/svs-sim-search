import sys
import pandas as pd
import requests as r
import multiprocessing as mp
from redis.exceptions import ConnectionError
from time import sleep, perf_counter
from pathlib import Path
import app.shared_context as sc
from app.helper import create_index, slice_dataframe


input_path = Path(__file__).parent.parent / "input"
images_path = Path(__file__).parent.parent / "images"


def download(idx, sdf):
    dff = pd.read_json(sdf, orient="split")
    print(f"{idx}- df shape: {dff.shape}")

    for i, row in dff.iterrows():
        list_id = row['id']
        image_url = row['image']
        # downloading image
        img_data = r.get(image_url).content
        with open(images_path / f"image_{list_id}.jpg", "wb") as handler:
            handler.write(img_data)
        print(f"...{i} - image downloaded | {list_id}...")


def parallel_download(only_missing=False):
    cores = mp.cpu_count()
    df = pd.read_csv(input_path / "electronics_250k.csv")
    print(f"original shape: {df.shape}")
    if only_missing:
        downloaded_images = [int(image_path.name[6:-4]) for image_path in images_path.glob("*.jpg")]
        df = df[~df["id"].isin(downloaded_images)]
        print(f"modified shape: {df.shape}")
    data_frames = slice_dataframe(df, cores)

    procs = []
    start_time = perf_counter()
    for core in range(cores):
        serialized_df = data_frames[core].to_json(orient="split")
        proc = mp.Process(target=download, args=(core, serialized_df))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    end_time = perf_counter()
    total_time = end_time - start_time
    print(f"Process took {total_time:.4f} seconds")


def run():
    queue_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    queue_name = f"{sc.QUEUE_IMG}_{queue_id}"
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
        except (UnicodeDecodeError, AttributeError, ValueError) as exp:
            sc.api_logger.info(f"...ERROR - {type(exp)} | {exp}...")
            sc.api_logger.info(f"unicode-decode error detected - skipping")
            sleep(0.5)
            continue
        if decoded_data == sc.START_TOKEN:
            sc.api_logger.info("start token detected")
            sleep(0.5)
            continue
        if decoded_data == sc.END_TOKEN:
            sc.api_logger.info("end token detected")
            sc.api_logger.info(f"{num_embeddings} embeddings inserted")
            sc.api_logger.info("waiting other processes finish")
            sleep(30)
            if queue_id == sc.QUEUE_MAIN:
                sc.api_logger.info("creating index on redis")
                create_index(
                    index_name="idx_img",
                    distance_metric=sc.IMG_DISTANCE_METRIC,
                    vector_field_name="embedding",
                    embedding_dimension=sc.IMG_EMBEDDING_DIMENSION,
                    # number_of_vectors=num_embeddings,
                    index_type="HNSW",
                    prefix="img::"
                )
                sc.api_logger.info("erasing stream")
                stream_group = sc.api_redis_cli.xread(streams={queue_name: 0})
                for streams in stream_group:
                    stream_name, messages = streams
                    [sc.api_redis_cli.xdel(stream_name, i[0]) for i in messages]
            break
        key, sentence = decoded_data.split(sc.SEPARATOR)
        filename = f"image_{key}.jpg"
        try:
            embeddings = sc.encode_image(img_path=images_path / filename)
        except FileNotFoundError as exp:
            print(f"{exp}")
            continue
        sc.api_logger.info(f"key: {key} | embeddings shape: {embeddings.shape}")
        embeddings_bytes = embeddings.detach().numpy().astype(sc.IMG_EMBEDDING_TYPE).tobytes()
        # bucket = int(key) % sc.BUCKETS
        sc.api_redis_cli.hset(
            f"img::{key}",
            mapping={
                "embedding": embeddings_bytes,
                "id": key
            }
        )
        num_embeddings += 1


if __name__ == '__main__':
    sc.api_redis_cli = sc.start_queueing(manually=True)
    sc.api_logger = sc.start_encoder_logging()
    run()
    # parallel_download()
