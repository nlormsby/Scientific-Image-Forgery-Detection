"""
Script to unify BioFors and Kaggle (Recod.ai) datasets into a single standardized format.

Kaggle Dataset:
- Images: train_images/forged/*.png and train_images/authentic/*.png
- Masks: train_masks/*.npy (only for forged images)
- Mask format: NumPy arrays with shape (1, H, W) or (2, H, W), values 0 or 1

BioFors Dataset:
- Images: biofors_images/<folder_id>/<image_id>.png
- Masks: annotation_files/idd_gt.json (coordinate format: [x1, y1, x2, y2])
- All images are forged (no authentic images)

Unified Structure:
Datasets/unified/
├── train/
│   ├── images/
│   └── masks/  (binary PNG masks, 0=authentic, 255=forged)
├── val/
│   ├── images/
│   └── masks/
└── metadata.csv
"""

import os
import json
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_directories():
    """Create unified dataset directory structure."""
    dirs = [
        "Datasets/unified/train/images",
        "Datasets/unified/train/masks",
        "Datasets/unified/val/images",
        "Datasets/unified/val/masks",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Created unified dataset directories")

def process_kaggle_dataset():
    """
    Process Kaggle (Recod.ai) dataset.
    Returns list of (image_path, mask_path, is_forged, source, image_type)
    """
    data_list = []

    # Process forged images
    forged_dir = Path("Datasets/train_images/forged")
    mask_dir = Path("Datasets/train_masks")

    forged_images = list(forged_dir.glob("*.png"))
    print(f"\nProcessing Kaggle forged images: {len(forged_images)}")

    for img_path in tqdm(forged_images):
        mask_path = mask_dir / f"{img_path.stem}.npy"
        if mask_path.exists():
            data_list.append({
                'image_path': str(img_path),
                'mask_path': str(mask_path),
                'is_forged': True,
                'source': 'kaggle',
                'image_type': 'scientific_figure',  # General scientific figures
                'mask_format': 'npy'
            })

    # Process authentic images
    authentic_dir = Path("Datasets/train_images/authentic")
    authentic_images = list(authentic_dir.glob("*.png"))
    print(f"Processing Kaggle authentic images: {len(authentic_images)}")

    for img_path in tqdm(authentic_images):
        data_list.append({
            'image_path': str(img_path),
            'mask_path': None,  # Authentic images have no mask
            'is_forged': False,
            'source': 'kaggle',
            'image_type': 'scientific_figure',
            'mask_format': None
        })

    # Process supplemental images if they exist
    supp_img_dir = Path("Datasets/supplemental_images")
    supp_mask_dir = Path("Datasets/supplemental_masks")

    if supp_img_dir.exists():
        supp_images = list(supp_img_dir.glob("*.png"))
        print(f"Processing Kaggle supplemental images: {len(supp_images)}")

        for img_path in tqdm(supp_images):
            mask_path = supp_mask_dir / f"{img_path.stem}.npy"
            if mask_path.exists():
                data_list.append({
                    'image_path': str(img_path),
                    'mask_path': str(mask_path),
                    'is_forged': True,
                    'source': 'kaggle_supplemental',
                    'image_type': 'scientific_figure',
                    'mask_format': 'npy'
                })

    print(f"✓ Total Kaggle samples: {len(data_list)}")
    return data_list

def process_biofors_dataset():
    """
    Process BioFors dataset.
    Returns list of (image_path, mask_coords, is_forged, source, image_type)
    """
    data_list = []

    # Load annotations
    idd_gt_path = Path("Datasets/annotation_files/idd_gt.json")
    classification_path = Path("Datasets/annotation_files/classification.json")

    with open(idd_gt_path, 'r') as f:
        idd_gt = json.load(f)

    with open(classification_path, 'r') as f:
        classification = json.load(f)

    biofors_root = Path("Datasets/biofors_images")

    print(f"\nProcessing BioFors dataset: {len(idd_gt)} folders")

    for folder_id, images_dict in tqdm(idd_gt.items()):
        folder_path = biofors_root / folder_id

        if not folder_path.exists():
            continue

        for image_name, coords_list in images_dict.items():
            image_path = folder_path / image_name

            if not image_path.exists():
                continue

            # Get image type from classification
            image_type = classification.get(folder_id, {}).get(image_name, "Unknown")

            data_list.append({
                'image_path': str(image_path),
                'mask_coords': coords_list,  # List of [x1, y1, x2, y2] boxes
                'is_forged': True,
                'source': 'biofors',
                'image_type': image_type,
                'mask_format': 'coordinates'
            })

    print(f"✓ Total BioFors samples: {len(data_list)}")
    return data_list

def convert_mask_to_png(mask_data, mask_format, image_shape, output_path):
    """
    Convert mask from various formats to binary PNG.

    Args:
        mask_data: Either numpy array (for Kaggle) or list of coordinates (for BioFors)
        mask_format: 'npy' or 'coordinates'
        image_shape: (H, W) of the corresponding image
        output_path: Where to save the PNG mask
    """
    H, W = image_shape[:2]

    if mask_format == 'npy':
        # Kaggle masks are numpy arrays
        mask = np.load(mask_data)

        # Handle different mask shapes
        if len(mask.shape) == 3:
            if mask.shape[0] == 1:
                # Shape: (1, H, W) -> (H, W)
                mask = mask[0]
            elif mask.shape[0] == 2:
                # Shape: (2, H, W) -> combine both channels
                mask = np.logical_or(mask[0], mask[1]).astype(np.uint8)
            else:
                # Take first channel
                mask = mask[0]

        # Ensure binary and correct dtype
        mask = (mask > 0).astype(np.uint8) * 255

    elif mask_format == 'coordinates':
        # BioFors masks are bounding boxes
        mask = np.zeros((H, W), dtype=np.uint8)

        for coords in mask_data:
            x1, y1, x2, y2 = coords
            # Fill the bounding box region with 255 (forged)
            mask[y1:y2, x1:x2] = 255

    else:
        raise ValueError(f"Unknown mask format: {mask_format}")

    # Resize if needed to match image size
    if mask.shape[0] != H or mask.shape[1] != W:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Save as PNG
    cv2.imwrite(output_path, mask)

def create_unified_dataset(val_split=0.2, random_seed=42):
    """
    Create unified dataset by combining Kaggle and BioFors.

    Args:
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
    """
    print("="*60)
    print("CREATING UNIFIED DATASET")
    print("="*60)

    # Create directories
    create_directories()

    # Process both datasets
    kaggle_data = process_kaggle_dataset()
    biofors_data = process_biofors_dataset()

    # Combine
    all_data = kaggle_data + biofors_data
    print(f"\n✓ Total samples: {len(all_data)}")
    print(f"  - Forged: {sum(1 for d in all_data if d['is_forged'])}")
    print(f"  - Authentic: {sum(1 for d in all_data if not d['is_forged'])}")

    # Split into train/val
    train_data, val_data = train_test_split(
        all_data,
        test_size=val_split,
        random_state=random_seed,
        stratify=[d['is_forged'] for d in all_data]  # Stratify by forged/authentic
    )

    print(f"\n✓ Train samples: {len(train_data)}")
    print(f"✓ Val samples: {len(val_data)}")

    # Process train set
    print("\n" + "="*60)
    print("PROCESSING TRAIN SET")
    print("="*60)
    metadata_train = process_split(train_data, 'train')

    # Process val set
    print("\n" + "="*60)
    print("PROCESSING VAL SET")
    print("="*60)
    metadata_val = process_split(val_data, 'val')

    # Save metadata
    metadata_all = pd.concat([metadata_train, metadata_val], ignore_index=True)
    metadata_all.to_csv("Datasets/unified/metadata.csv", index=False)
    print(f"\n✓ Saved metadata to Datasets/unified/metadata.csv")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(metadata_all.groupby(['split', 'is_forged', 'source']).size().unstack(fill_value=0))
    print(f"\n✓ Dataset unification complete!")

def process_split(data_list, split_name):
    """
    Process a single split (train or val).

    Args:
        data_list: List of data dictionaries
        split_name: 'train' or 'val'

    Returns:
        DataFrame with metadata
    """
    metadata_rows = []

    for idx, data in enumerate(tqdm(data_list)):
        # Generate unique filename
        unified_id = f"{split_name}_{idx:06d}"

        # Load image to get dimensions
        image = cv2.imread(data['image_path'])
        if image is None:
            print(f"Warning: Could not read {data['image_path']}")
            continue

        H, W = image.shape[:2]

        # Copy image to unified location
        output_image_path = f"Datasets/unified/{split_name}/images/{unified_id}.png"
        cv2.imwrite(output_image_path, image)

        # Process mask
        if data['is_forged']:
            output_mask_path = f"Datasets/unified/{split_name}/masks/{unified_id}.png"

            if data['mask_format'] == 'npy':
                convert_mask_to_png(
                    data['mask_path'],
                    'npy',
                    (H, W),
                    output_mask_path
                )
            elif data['mask_format'] == 'coordinates':
                convert_mask_to_png(
                    data['mask_coords'],
                    'coordinates',
                    (H, W),
                    output_mask_path
                )
        else:
            # Authentic images get all-zero masks
            output_mask_path = f"Datasets/unified/{split_name}/masks/{unified_id}.png"
            blank_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.imwrite(output_mask_path, blank_mask)

        # Record metadata
        metadata_rows.append({
            'unified_id': unified_id,
            'split': split_name,
            'original_path': data['image_path'],
            'is_forged': data['is_forged'],
            'source': data['source'],
            'image_type': data.get('image_type', 'Unknown'),
            'height': H,
            'width': W
        })

    return pd.DataFrame(metadata_rows)

if __name__ == "__main__":
    create_unified_dataset(val_split=0.2, random_seed=42)
