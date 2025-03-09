import numpy as np
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def set_all_seeds(seed): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEFAULT_TRANSFORMS = A.Compose([
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, p=0.3),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1, p=0.8),
    A.ToGray(p=0.1),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Affine(rotate=(-20, 20), scale=(0.9, 1.1), shear=(-10, 10), p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50, p=0.2),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])



def get_transform(mean, std, mode='train', crop_size=224):

    if mode == 'train':
        transform = A.Compose([
            A.Resize(crop_size, crop_size),
            A.RandomResizedCrop(
                size=(crop_size, crop_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.8
            ),
            DEFAULT_TRANSFORMS,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif mode == 'eval':
        transform = A.Compose([
            A.Resize(crop_size, crop_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f'Unknown transform mode: {mode}')

    return lambda x: transform(image=np.array(x))['image']


def get_dataloaders(cfg): 
    train_transform = get_transform(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], mode='train')
    eval_transform = get_transform(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], mode='eval')
    # 'datasets/DDR/train'
    train_ds = ImageFolder(cfg.train.dir, transform=train_transform)
    valid_ds = ImageFolder(cfg.val.dir, transform=eval_transform)
    test_ds = ImageFolder(cfg.test.dir, transform=eval_transform)
    
    train_loader = DataLoader(
        train_ds, shuffle=True,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers, 
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_ds, shuffle=False,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds, shuffle=False,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        pin_memory=False
    )
    return train_loader, valid_loader, test_loader
