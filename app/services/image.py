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


# model name: "mobilenet_v2"
# pool strategy: "mean"
# sample execution (requires torchvision)
#
# from PIL import Image
# from torchvision import transforms
#
#
# input_image = Image.open(filename)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
#
# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')
#
# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)
