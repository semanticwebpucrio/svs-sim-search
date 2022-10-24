import pandas as pd
from pathlib import Path
import requests as r


input_path = Path.cwd() / "input"
images_path = Path.cwd() / "images"


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


if __name__ == '__main__':
    main()
