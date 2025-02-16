import json
import os

from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch

size_cond = {'shortest_edge': 224}
preprocess = CLIPImageProcessor(crop_size={'height': 224, 'width': 224},
                                do_center_crop=True,
                                do_convert_rgb=True,
                                do_normalize=True,
                                do_rescale=True,
                                do_resize=True,
                                image_mean=[0.48145466, 0.4578275, 0.40821073],
                                image_std=[0.26862954, 0.26130258, 0.27577711],
                                resample=3,
                                size=size_cond,
                                )

image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",
                                                              torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_encoder.to(device)

dataset_root = "C:\\Users\\user\\KDBC2024\\dataset\\fashion_iq\\"
img_data_root = os.path.join(dataset_root, 'images')
with open(
        os.path.join(dataset_root, 'captions', 'cap.{}.{}.json'.format("dress", "train"))) as json_file:
    img_caption_data = json.load(json_file)

img_caption_pair = img_caption_data[0]
key = 'candidate'
img_path = os.path.join(img_data_root, '{}.jpg'.format(img_caption_pair[key]))
print(img_path)

with open(img_path, 'rb') as f:
    img = Image.open(f).convert('RGB')
    processed_img = preprocess(img)
    print(processed_img['pixel_values'][0])
    img_tensor = torch.from_numpy(processed_img['pixel_values'][0])
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

img_features = image_encoder(img_tensor.to(image_encoder.dtype))
print(img_features)
print(img_features.image_embeds.float())
