import json
# import cv2
import numpy as np
from PIL import Image, ImageOps

from torch.utils.data import Dataset
import random


class COCODataset_train(Dataset):
    def __init__(self):
        with open("/path/to/new_train_coco_train_test.json", 'rt') as f:
            for line in f:
                self.data = json.loads(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]


        image_filename = item['image_file_name']
        category_name = item['category_names']

        if len(category_name) == 0:
            category_name.append("others")
        else:
            selected_name = random.choice(category_name)
            category_name = [selected_name]

        image = Image.open('/path/to/dataset/train2017/' + image_filename).convert('RGB')
        image = image.resize((256, 256))

        image = np.array(image)
        image = image.astype(np.float32) / 255.0

        # return dict(image=image, labels=list(set(category_name)))
        return dict(image=image, labels=category_name)

class COCODataset_test(Dataset):
    def __init__(self):
        # self.data = []
        with open("/path/to/val_test_select.json", 'rt') as f:
            for line in f:
                self.data = json.loads(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_filename = item['image_file_name']
        category_name = item['category_names']


        if len(category_name) == 0:
            category_name.append("others")
        else:
            selected_name = random.choice(category_name)
            category_name = [selected_name]



        image = Image.open("/path/to/dataset/val2017/" + image_filename).convert('RGB')
        image = image.resize((256, 256))


        # Normalize source images to [0, 1].
        image = np.array(image)
        image = image.astype(np.float32) / 255.0



        return dict(image=image, labels=category_name, name=image_filename)