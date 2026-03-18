"""
Dataset class and augmentation functions for forgery detection.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ForgeryDataset(Dataset):
    """
    Dataset for forgery detection.

    Structure:
        images_dir/
            image_0001.png
            image_0002.png
            ...
        masks_dir/
            image_0001.png
            image_0002.png
            ...
    """

    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        """
        Args:
            images_dir: Path to directory with images
            masks_dir: Path to directory with masks
            augmentation: Albumentations augmentation pipeline
            preprocessing: Albumentations preprocessing pipeline
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.png')))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image filename
        image_path = self.image_files[idx]
        mask_path = self.masks_dir / image_path.name

        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Apply augmentations (don't add channel dimension yet)
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Apply preprocessing
        if self.preprocessing:
            preprocessed = self.preprocessing(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask']

        # Add channel dimension to mask if it doesn't have one
        # After ToTensorV2, image is (C, H, W) and mask should be (1, H, W)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)

        return {
            'image': image,
            'mask': mask,
            'filename': image_path.name
        }


def get_augmentation(mode='train'):
    """
    Get augmentation pipeline.

    Args:
        mode: 'train' or 'val'

    Returns:
        Albumentations composition
    """
    if mode == 'train':
        return A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),

            # Resize to fixed size
            A.Resize(512, 512),

            # Color/intensity transforms (only on image, not mask)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
                A.CLAHE(clip_limit=2.0, p=1),
            ], p=0.5),

            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.3),

            # Normalize (per channel)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),

            ToTensorV2(),
        ])
    else:  # validation
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])


def get_preprocessing():
    """
    Get preprocessing pipeline (applied after augmentation).

    Returns:
        Albumentations composition or None
    """
    # Preprocessing is now included in augmentation pipeline
    return None
