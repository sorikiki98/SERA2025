import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import os
import json


def _create_modifier_from_attributes(ref_attribute, targ_attribute):
    return ref_attribute + " and " + targ_attribute


class FashionIQDataset(Dataset):
    def __init__(self, data_root, tokenizer, image_encoder, image_processor, size, placeholder_img_token, device,
                 clothing_type='dress', repeats=1, center_crop=False, split='train'):
        self.data_root = data_root
        self.img_data_root = os.path.join(self.data_root, 'images')
        self.placeholder_img_token = placeholder_img_token
        self.placeholder_new_tokens = None
        self.clothing_type = clothing_type
        assert self.clothing_type in {'dress', 'toptee', 'shirt'}, (
            f"Invalid clothing type '{self.clothing_type}'. "
        )
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder
        '''
        for name, param in self.image_encoder.named_parameters():
            print(f"Layer: {name}, Mean: {param.data.mean().item()}, Std: {param.data.std().item()}")
        '''
        self.image_processor = image_processor
        self.size = size
        self.repeats = repeats
        self.center_crop = center_crop
        self.split = split
        self.device = device

        def get_img_caption_json(dataset_root, clothing_type):
            with open(
                    os.path.join(dataset_root, 'captions', 'cap.{}.{}.json'.format(clothing_type, split))) as json_file:
                img_caption_data = json.load(json_file)
            '''
            new_tokens = list()
            # 1. check if some captions include new tokens
            for item in img_caption_data:
                captions = item['captions']

                ref_caption = captions[0]
                targ_caption = captions[1]

                ref_new_tokens = self.get_new_tokens_per_caption(ref_caption)
                targ_new_tokens = self.get_new_tokens_per_caption(targ_caption)

                new_tokens.extend(ref_new_tokens)
                new_tokens.extend(targ_new_tokens)
                new_tokens = list(set(new_tokens))
            # 2. add a placeholder object token
            self.placeholder_new_tokens = new_tokens
            if len(new_tokens) == 0:
                print("No tokens has been added among our image captions!")
            '''
            return img_caption_data

        self.img_caption_data = get_img_caption_json(data_root, self.clothing_type)

    def __getitem__(self, idx):
        safe_idx = idx // 2
        reverse = (idx % 2 == 1)

        ref_img_path, _ = self._get_img_path_using_idx(safe_idx, is_ref=True)

        targ_img_path, _ = self._get_img_path_using_idx(safe_idx, is_ref=False)
        ref_img_pixel_values = self.get_img_pixel_values_from_path(ref_img_path)
        # targ_img_pixel_values = self.get_img_pixel_values_from_path(targ_img_path)

        ref_text, target_text = self._get_modifier(safe_idx, reverse=reverse)
        text_with_img_token = self.placeholder_img_token + " " + ref_text

        img = dict()
        img['pixel_values'] = ref_img_pixel_values
        img['text'] = text_with_img_token

        ref_img_pixel_values = ref_img_pixel_values.unsqueeze(0)

        with torch.no_grad():
            img_features = self.image_encoder(ref_img_pixel_values.to(self.image_encoder.dtype))

        img['image_embeds'] = img_features.image_embeds.squeeze(0).float()
        img['input_ids_placeholder_img'] = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(
                self.placeholder_img_token))
        img['input_ids_placeholder_new'] = torch.tensor(-1)  # todo
        img['input_ids'] = self.tokenizer(
            img['text'],
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        ).input_ids[0]

        return img

    def __len__(self):
        return len(self.img_caption_data) * 2

    def _get_img_set_json_as_list(self, clothing_type, set):
        with open(os.path.join(self.data_root, 'image_sets', 'set.{}.{}.json'.format(clothing_type, set))) as json_file:
            img_set_list = json.load(json_file)
        return img_set_list

    def _create_img_path_from_id(self, id):
        return os.path.join(self.img_data_root, '{}.jpg'.format(id))

    def _get_modifier(self, idx, reverse=False):
        img_caption_pair = self.img_caption_data[idx]
        cap1, cap2 = img_caption_pair['captions']
        return cap1, cap2
        # return _create_modifier_from_attributes(cap1, cap2) if not reverse else _create_modifier_from_attributes(cap2, cap1)

    def _get_img_path_using_idx(self, idx, is_ref=True):
        img_caption_pair = self.img_caption_data[idx]
        key = 'candidate' if is_ref else 'target'

        img = self._create_img_path_from_id(img_caption_pair[key])
        id = img_caption_pair[key]

        return img, id

    def get_img_pixel_values_from_path(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            # print("img", img)
            processed_img = self.image_processor(img)
            img_tensor = torch.from_numpy(processed_img['pixel_values'][0])
            # print("processed_img", img_tensor.size())

        return img_tensor
