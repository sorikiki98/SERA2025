import numpy as np
import PIL.Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import random


class Fashion200kAttribute(Dataset):
    def __init__(self, path, split='train', n_same_children=3, transform=None):
        super(Fashion200kAttribute, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + '/'
        self.n_same_children = n_same_children

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]
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
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                }
                self.imgs += [img]

        self.init_parent2children_captions()
        self.sample_children_having_the_most_siblings()

    def __len__(self):
        return len(self.filtered_imgs)

    def __getitem__(self, idx):
        attr = self.filtered_imgs[idx]['attr']
        child_image = self.get_img(idx)

        return self.parent_prompt, child_image, attr

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
        self.parent2children_captions = parent2children_captions

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
            self.parent_prompt = p
            for c in children:
                _, attr = self.get_different_word(p, c)
                if len(self.caption2imgids[c]) >= self.n_same_children:
                    child_images[attr] = []
                    child_image_ids = random.sample(self.caption2imgids[c], k=self.n_same_children)
                    child_images[attr].extend(child_image_ids)

        filtered_imgs = []
        for attr in child_images:
            for img_id in child_images[attr]:
                chosen_image = self.imgs[img_id]
                chosen_image['attr'] = attr
                chosen_image['parent_attr'] = self.parent_prompt
                filtered_imgs.append(chosen_image)
        self.filtered_imgs = filtered_imgs

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word = self.get_different_word(source_caption, target_caption)
        return idx, target_idx, source_word, target_word

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.filtered_imgs[idx]['file_path']
        img_path = img_path.replace('/', '\\')
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img
