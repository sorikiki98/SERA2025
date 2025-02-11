import random
from pathlib import Path
from typing import Dict, Any, List, Union

import PIL
import numpy as np
import torch
from PIL import Image
from packaging import version
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from constants import IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_SMALL

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


class Fashion200K(Dataset):

    def __init__(self, data_root: str,
                 tokenizer: CLIPTokenizer,
                 fixed_object_token_or_path: Union[Path, str] = None,
                 size: int = 512,
                 repeats: int = 100,
                 interpolation: str = "bicubic",
                 flip_p: float = 0.5,
                 set: str = "train",
                 placeholder_object_token: str = "*",
                 center_crop: bool = False):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_object_token = placeholder_object_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = IMAGENET_STYLE_TEMPLATES_SMALL
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # get label files for the split
        label_path = self.data_root + 'labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if 'train' in f]
        # read image info from label files
        self.imgs = []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            with open(label_path + '\\' + filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])]
                }
                self.imgs += [img]

        self.parent2children_captions = self.init_parent2children_captions()
        self.filtered_imgs = self.sample_children_having_the_most_siblings()
        self.num_images = len(self.filtered_imgs)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.fixed_object_token = fixed_object_token_or_path
        self.fixed_object_token_pretrained = False
        self.placeholder_object_tokens = [self.fixed_object_token]
        self.placeholder_style_tokens = self.generate_style_tokens()

        self.placeholder_tokens = self.placeholder_style_tokens + self.placeholder_object_tokens
        assert hasattr(self, 'placeholder_object_tokens')
        assert hasattr(self, 'placeholder_style_tokens')

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = i % self.num_images

        placeholder_object_token = self.placeholder_object_tokens[0]
        assert len(self.placeholder_object_tokens) == 1

        img = self.filtered_imgs[idx]
        attr = img['attr']
        img['image_idx'] = idx
        style_token = f'<style_{attr}>'
        img['pixel_values'] = self.get_pixel_values(idx)
        img['text'] = random.choice(self.templates).format(self.fixed_object_token, style_token)

        if self.fixed_object_token_pretrained:
            img['input_ids_placeholder_object'] = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(
                    placeholder_object_token))
        else:
            img['input_ids_placeholder_object'] = torch.tensor(-1)
        img['input_ids_placeholder_style'] = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(style_token))
        img['input_ids'] = self.tokenizer(
            img['text'],
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        ).input_ids[0]

        return img

    def get_filtered_different_words(self):
        words = [img['attr'] for img in self.filtered_imgs]
        return set(words)

    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()

        source_word = ''
        target_word = ''

        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        return source_word, target_word

    def init_parent2children_captions(self):
        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        return parent2children_captions

    def sample_children_having_the_most_siblings(self):
        parent2children_captions = self.parent2children_captions
        num_parent2children = []
        for p in parent2children_captions:
            num_parent2children.append(len(parent2children_captions[p]))

        max_num_parent2children = max(num_parent2children)
        max_parent2children = {p: parent2children_captions[p] for p in parent2children_captions if
                               len(parent2children_captions[p]) == max_num_parent2children}

        child_images = {}  # 속성: [이미지 id, 이미지 id]
        for p, children in max_parent2children.items():
            for c in children:
                _, attr = self.get_different_word(p, c)
                if len(self.caption2imgids[c]) >= 2:
                    child_images[attr] = []
                    child_image_ids = random.sample(self.caption2imgids[c], k=2)
                    child_images[attr].extend(child_image_ids)

        filtered_imgs = []
        for attr in child_images:
            for img_id in child_images[attr]:
                chosen_image = self.imgs[img_id]
                chosen_image['attr'] = attr
                filtered_imgs.append(chosen_image)

        return filtered_imgs

    def get_pixel_values(self, idx, raw_img=False):
        image_path = self.data_root + self.filtered_imgs[idx]['file_path']
        image_path = image_path.replace('/', '\\')
        with open(image_path, 'rb') as f:
            image = PIL.Image.open(f)
            image = image.convert('RGB')
        if raw_img:
            return image

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)

        return image

    def generate_style_tokens(self):
        placeholder_style_tokens = ['<style_' + img['attr'] + '>' for img in self.filtered_imgs]
        placeholder_style_tokens = list(set(placeholder_style_tokens))

        return placeholder_style_tokens
