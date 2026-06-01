import torch
import random
import numpy as np
import ttach as tta
import albumentations as A

from scipy import ndimage
from scipy.ndimage.interpolation import zoom


class random_generator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, image, label):

        if random.random() > 1:
            image, label = self.random_rot_flip(image, label)
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
        
        h, w = image.shape[1:]
        label_h, label_w = label.shape
        # if h != self.output_size[0] or w != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
        #     label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        
        low_res_label = torch.tensor(low_res_label, dtype=torch.float32)
        image, mask = np.array(image), np.array(low_res_label.long())
        return image, mask
    
    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    
    def random_rotate(self, image, label):
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


class data_augmentation(object):
    def __init__(self, mode='train'):
        self.mode = mode
        self.train_transform = A.Compose([
            A.Resize(1024, 1024),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], p=0.2),
            A.OneOf([
                A.Rotate(limit=90, interpolation=1, border_mode=2, p=1.0),
                A.ShiftScaleRotate(shift_limit=(-0.1, 0.1), scale_limit=(0.9, 1.1), rotate_limit=(-45, 45), interpolation=1, border_mode=0, p=1.0)
            ], p=0.5),
            A.CoarseDropout(max_holes=64, max_height=64, max_width=64, min_holes=0, min_height=16, min_width=16, fill_value=0, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.6),
            A.RandomRotate90(p=0.25),
            A.Normalize(),
        ]

    def __call__(self, image, mask):
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            image, mask = np.array(image), np.array(mask)
            

        if self.mode == 'train':
            transformed = self.train_transform(image=image.copy(), mask=mask.copy())
        elif self.mode == 'val':
            transformed = self.val_transform(image=image.copy(), mask=mask.copy())
        else:
            raise ValueError("Invalid mode. Mode must be 'train' or 'val'.")

        image, mask = transformed['image'], transformed['mask']
        return image, mask
