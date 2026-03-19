"""
Create Proper Train/Val/Test Split (70/15/15)

This script takes the existing unified dataset and re-splits it into:
- Train: 70% (for training)
- Val: 15% (for hyperparameter tuning and model selection)
- Test: 15% (for final unbiased evaluation)

Usage:
1. Backup your current data: cp -r Datasets/unified Datasets/unified_backup
2. Run: python create_proper_split.py

This will:
- Read metadata.csv
- Split data 70/15/15 (stratified by is_forged)
- Update metadata.csv with new splits
- Move files to train/val/test directories
"""

import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# Configuration
DATA_DIR = Path("Datasets/unified")
METADATA_PATH = DATA_DIR / "metadata.csv"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

def create_three_way_split():
    """Create proper train/val/test split from existing data."""

    print("="*70)
    print("CREATING PROPER TRAIN/VAL/TEST SPLIT")
    print("="*70)
    print(f"Split ratios: Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%} / Test {TEST_RATIO:.0%}")
    print(f"Random seed: {RANDOM_SEED}\n")

    # Read metadata
    print("Reading metadata...")
    metadata = pd.read_csv(METADATA_PATH)

    print(f"Total samples: {len(metadata)}")
    print(f"Forged samples: {metadata['is_forged'].sum()}")
    print(f"Authentic samples: {(~metadata['is_forged']).sum()}\n")

    # First split: separate train from (val + test)
    # Stratify by is_forged to maintain class balance
    train_df, val_test_df = train_test_split(
        metadata,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=metadata['is_forged']
    )

    # Second split: separate val from test
    # Adjust test_size to get correct val/test ratio
    val_test_split_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=val_test_split_ratio,
        random_state=RANDOM_SEED,
        stratify=val_test_df['is_forged']
    )

    # Update split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Combine back together
    new_metadata = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Print split statistics
    print("New split distribution:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(metadata)*100:.1f}%)")
    print(f"    - Forged: {train_df['is_forged'].sum()}")
    print(f"    - Authentic: {(~train_df['is_forged']).sum()}")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(metadata)*100:.1f}%)")
    print(f"    - Forged: {val_df['is_forged'].sum()}")
    print(f"    - Authentic: {(~val_df['is_forged']).sum()}")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(metadata)*100:.1f}%)")
    print(f"    - Forged: {test_df['is_forged'].sum()}")
    print(f"    - Authentic: {(~test_df['is_forged']).sum()}\n")

    # Ask for confirmation
    print("This will reorganize your dataset files.")
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return

    print("\nReorganizing files...")

    # Create new directory structure
    test_images_dir = DATA_DIR / "test" / "images"
    test_masks_dir = DATA_DIR / "test" / "masks"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_masks_dir.mkdir(parents=True, exist_ok=True)

    # Track files to move
    moves = {
        'train': {'count': 0, 'unified_ids': []},
        'val': {'count': 0, 'unified_ids': []},
        'test': {'count': 0, 'unified_ids': []}
    }

    # Process each sample
    for _, row in new_metadata.iterrows():
        old_split = metadata[metadata['unified_id'] == row['unified_id']]['split'].iloc[0]
        new_split = row['split']
        unified_id = row['unified_id']

        # If split changed, move files
        if old_split != new_split:
            # Image files
            old_img_path = DATA_DIR / old_split / "images" / f"{unified_id}.png"
            new_img_path = DATA_DIR / new_split / "images" / f"{unified_id}.png"

            # Mask files
            old_mask_path = DATA_DIR / old_split / "masks" / f"{unified_id}.png"
            new_mask_path = DATA_DIR / new_split / "masks" / f"{unified_id}.png"

            # Move image
            if old_img_path.exists():
                shutil.move(str(old_img_path), str(new_img_path))

            # Move mask
            if old_mask_path.exists():
                shutil.move(str(old_mask_path), str(new_mask_path))

        moves[new_split]['count'] += 1
        moves[new_split]['unified_ids'].append(unified_id)

    print(f"✓ Moved files to new splits")
    print(f"  Train: {moves['train']['count']} files")
    print(f"  Val:   {moves['val']['count']} files")
    print(f"  Test:  {moves['test']['count']} files\n")

    # Save updated metadata
    backup_path = METADATA_PATH.parent / "metadata_old.csv"
    shutil.copy(METADATA_PATH, backup_path)
    print(f"✓ Backed up old metadata to: {backup_path}")

    new_metadata.to_csv(METADATA_PATH, index=False)
    print(f"✓ Saved new metadata to: {METADATA_PATH}")

    # Verify file counts
    print("\nVerifying file counts...")
    for split in ['train', 'val', 'test']:
        images_dir = DATA_DIR / split / "images"
        masks_dir = DATA_DIR / split / "masks"

        num_images = len(list(images_dir.glob("*.png")))
        num_masks = len(list(masks_dir.glob("*.png")))
        expected = moves[split]['count']

        print(f"  {split.capitalize():5s}: {num_images} images, {num_masks} masks (expected: {expected})")

        if num_images != expected or num_masks != expected:
            print(f"    ⚠️  Warning: File count mismatch!")

    print("\n" + "="*70)
    print("SPLIT CREATION COMPLETE!")
    print("="*70)
    print(f"New dataset structure:")
    print(f"  Datasets/unified/")
    print(f"    ├── train/          ({moves['train']['count']} samples)")
    print(f"    ├── val/            ({moves['val']['count']} samples)")
    print(f"    ├── test/           ({moves['test']['count']} samples)")
    print(f"    └── metadata.csv    (updated)")
    print(f"\nOld metadata backed up to: {backup_path}")
    print("="*70)

    # Save split statistics
    stats_path = DATA_DIR / "split_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DATASET SPLIT STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        f.write(f"Total samples: {len(metadata)}\n\n")
        f.write("Split distribution:\n")
        f.write(f"  Train: {len(train_df)} ({len(train_df)/len(metadata)*100:.1f}%)\n")
        f.write(f"    - Forged: {train_df['is_forged'].sum()}\n")
        f.write(f"    - Authentic: {(~train_df['is_forged']).sum()}\n\n")
        f.write(f"  Val:   {len(val_df)} ({len(val_df)/len(metadata)*100:.1f}%)\n")
        f.write(f"    - Forged: {val_df['is_forged'].sum()}\n")
        f.write(f"    - Authentic: {(~val_df['is_forged']).sum()}\n\n")
        f.write(f"  Test:  {len(test_df)} ({len(test_df)/len(metadata)*100:.1f}%)\n")
        f.write(f"    - Forged: {test_df['is_forged'].sum()}\n")
        f.write(f"    - Authentic: {(~test_df['is_forged']).sum()}\n\n")
        f.write("="*70 + "\n")

    print(f"\n✓ Split statistics saved to: {stats_path}")


def main():
    print("\n⚠️  IMPORTANT: This script will reorganize your dataset files!")
    print("   Make sure you have a backup before proceeding.\n")
    print("   Recommended: cp -r Datasets/unified Datasets/unified_backup\n")

    if not METADATA_PATH.exists():
        print(f"❌ Error: Metadata file not found at {METADATA_PATH}")
        print(f"   Make sure you're running this from the project root directory.")
        return

    create_three_way_split()


if __name__ == '__main__':
    main()
