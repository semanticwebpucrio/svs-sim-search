import sys
import torch
import pandas as pd
import requests as r
from time import sleep
from pathlib import Path
import app.shared_context as sc
from app.helper import create_flat_index


input_path = Path(__file__).parent.parent / "input"
images_path = Path(__file__).parent.parent / "images"


def download(image_url, list_id=None):
    img_data = r.get(image_url).content
    if not list_id:
        return img_data
    with open(f"{images_path}/image_{list_id}.jpg", "wb") as handler:
        handler.write(img_data)


def main(filename="electronics_20220615_original.csv", skip=0):
    df = pd.read_csv(input_path / filename)
    print(f"dataframe: {df.shape}")
    for idx, row in df.iterrows():
        if int(str(idx)) < skip:
            continue
        list_id = row['id']
        image_url = row['image']
        download(list_id, image_url)
        print(f"{idx} - image downloaded | {list_id}")


def run():
    queue_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    queue_name = f"{sc.QUEUE_IMG}_{queue_id}"
    sc.api_logger.info("downloading cnn model")
    # pool_strategy="mean"
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", weights=True)
    model.eval()
    sc.api_logger.info(f"subscribing to redis queue: {queue_name}")
    p_img = sc.api_redis_cli.pubsub()
    p_img.subscribe(queue_name)

    sc.api_logger.info("starting loop")
    num_embeddings, num_empty_loops = 0, 0
    while True:
        msg = p_img.get_message()
        if not msg:
            sc.api_logger.info(f"empty {queue_name} - skipping #{num_empty_loops}")
            if num_embeddings > 0:
                num_empty_loops += 1
                if num_empty_loops >= sc.MAX_LOOPS_WITHOUT_DATA:
                    if queue_id == 0:  # only queue_id = 0 will be responsible to create index
                        sc.api_logger.info("creating index on redis")
                        create_flat_index(sc.IMG_EMBEDDING_FIELD_NAME, num_embeddings, index_name="idx_img")
                    num_embeddings, num_empty_loops = 0, 0
            sleep(0.5)
            continue
        try:
            key, _img_url = msg.get("data").decode().split(sc.SEPARATOR)
        except (UnicodeDecodeError, AttributeError, ValueError):
            sc.api_logger.info(f"unicode-decode error detected - skipping")
            sleep(0.5)
            continue

        filename = f"image_{key}.jpg"
        embeddings = sc.encode_image(img_path=images_path / filename)
        sc.api_logger.info(f"key: {key} | embeddings shape: {embeddings.shape}")
        embeddings_bytes = embeddings.detach().numpy().astype(sc.IMG_EMBEDDING_TYPE).tobytes()
        sc.api_redis_cli.hset(
            f"img-{key}", mapping={"embedding": embeddings_bytes, "id": key}
        )
        num_embeddings += 1


if __name__ == '__main__':
    sc.api_redis_cli = sc.start_queueing()
    sc.api_logger = sc.start_encoder_logging()
    run()
