import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json


def _get_img_set_json_as_list(dataset_root, clothing_type, set):
    with open(os.path.join(dataset_root, 'image_sets', 'set.{}.{}.json'.format(clothing_type, set))) as json_file:
        img_set_list = json.load(json_file)
    return img_set_list


def _create_img_path_from_id(root, id):
    return os.path.join(root, '{}.jpg'.format(id))


def _get_modifier(img_caption_data, idx, reverse=False):
    img_caption_pair = img_caption_data[idx]
    cap1, cap2 = img_caption_pair['captions']
    return cap1, cap2
    # return _create_modifier_from_attributes(cap1, cap2) if not reverse else _create_modifier_from_attributes(cap2, cap1)


def _create_modifier_from_attributes(ref_attribute, targ_attribute):
    return ref_attribute + " and " + targ_attribute


def _get_img_path_using_idx(img_caption_data, img_root, idx, is_ref=True):
    img_caption_pair = img_caption_data[idx]
    key = 'candidate' if is_ref else 'target'

    img = _create_img_path_from_id(img_root, img_caption_pair[key])
    id = img_caption_pair[key]

    return img, id


class FashionIQDataset(Dataset):
    def __init__(self, data_root, tokenizer, size, placeholder_object_token, fixed_object_token_or_path=None,
                 clothing_type='dress', repeats=1, center_crop=False, split='train'):
        self.data_root = data_root
        self.img_data_root = os.path.join(self.data_root, 'images')
        self.placeholder_object_token = placeholder_object_token
        self.fixed_object_token = fixed_object_token_or_path
        self.fixed_object_token_pretrained = False
        self.placeholder_new_tokens = None
        self.clothing_type = fixed_object_token_or_path if fixed_object_token_or_path is not None else clothing_type
        assert self.clothing_type in {'dress', 'toptee', 'shirt'}, (
            f"Invalid clothing type '{self.clothing_type}'. "
        )
        self.tokenizer = tokenizer
        self.size = size
        self.repeats = repeats
        self.center_crop = center_crop
        self.split = split

        def get_img_caption_json(dataset_root, clothing_type):
            with open(
                    os.path.join(dataset_root, 'captions', 'cap.{}.{}.json'.format(clothing_type, split))) as json_file:
                img_caption_data = json.load(json_file)

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
            new_tokens.append(self.placeholder_object_token)
            self.placeholder_new_tokens = new_tokens
            return img_caption_data

        self.img_caption_data = get_img_caption_json(data_root, self.clothing_type)

    def __getitem__(self, idx):
        safe_idx = idx // 2
        reverse = (idx % 2 == 1)

        if self.placeholder_object_tokens is not None:
            placeholder_object_token = self.placeholder_object_tokens[0]
            assert len(self.placeholder_object_tokens) == 1

        ref_img_path, _ = _get_img_path_using_idx(self.img_caption_data, self.img_data_root, safe_idx, is_ref=True)
        targ_img_path, _ = _get_img_path_using_idx(self.img_caption_data, self.img_data_root, safe_idx, is_ref=False)
        ref_img_pixel_values = self.get_img_pixel_values_from_path(ref_img_path)
        targ_img_pixel_values = self.get_img_pixel_values_from_path(targ_img_path)

        ref_text, target_text = _get_modifier(self.img_caption_data, safe_idx, reverse=reverse)

        img = dict()
        img['pixel_values'] = ref_img_pixel_values
        img['text'] = ref_text

        if self.fixed_object_token_pretrained:
            img['input_ids_placeholder_object'] = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(
                    placeholder_object_token))
        else:
            img['input_ids_placeholder_object'] = torch.tensor(-1)
        # img['input_ids_placeholder_new'] = torch.tensor(
        #    self.tokenizer.convert_tokens_to_ids(img['ref_new_tokens']))
        img['input_ids_placeholder_new'] = torch.tensor(-1)
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

    def get_img_pixel_values_from_path(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        img = np.array(img).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)

        return image

    def get_new_tokens_per_caption(self, text):
        '''
        encoded = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoded['input_ids']
        offsets = encoded['offset_mapping']
        oov_tokens = [text[start:end] for tid, (start, end) in zip(input_ids, offsets) if
                      tid == self.tokenizer.unk_token_id]
        '''
        oov_tokens = []
        return oov_tokens
