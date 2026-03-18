"""
Stage 4: Hyperparameter Tuning for FPN + ResNeXt50

Goal: Optimize the winning architecture from Phases 1-3
- Phase 1 winner: FPN architecture
- Phase 2 winner: ResNeXt50 backbone
- Phase 3 insight: Top-down pathway is critical

This script tests different hyperparameter combinations:
1. Loss functions (BCE, Dice, BCE+Dice, Focal, Tversky)
2. Learning rates (1e-3, 5e-4, 1e-4, 5e-5)
3. Augmentation strategies (basic, advanced, none)
4. Batch sizes (2, 4, 8)

Usage: python hyperparam_tuning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from dataset import ForgeryDataset, get_augmentation
from utils import AverageMeter, calculate_metrics, calculate_image_level_accuracy

# ============================================================================
# STAGE 4 CONFIGURATION
# ============================================================================

# Use the winning architecture from Phases 1-3
USE_CUSTOM_FPN = False  # Set to True to use custom FPN, False for SMP FPN

if USE_CUSTOM_FPN:
    from Models.fpn import create_fpn_full
    BASE_MODEL_FN = lambda: create_fpn_full(backbone_name='resnext50_32x4d')
    BASE_MODEL_NAME = "custom_fpn_resnext50"
else:
    BASE_MODEL_FN = lambda: smp.FPN(encoder_name='resnext50_32x4d', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)
    BASE_MODEL_NAME = "smp_fpn_resnext50"

# Data settings
DATA_DIR = "Datasets/unified"
MAX_SAMPLES = 1000  # Use 1000 for quick experiments, set to None for full dataset

# Training settings
EPOCHS = 15
NUM_WORKERS = 4
SEED = 42

# ============================================================================
# HYPERPARAMETER EXPERIMENTS
# ============================================================================

EXPERIMENTS = [
    # ========================================================================
    # Baseline - Current best configuration from Phases 1-3
    # ========================================================================
    {
        'name': 'baseline',
        'loss': 'bce_dice',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },

    # ========================================================================
    # Experiment 1: Loss Function Comparison
    # ========================================================================
    {
        'name': 'loss_bce',
        'loss': 'bce',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },
    {
        'name': 'loss_dice',
        'loss': 'dice',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },
    {
        'name': 'loss_focal',
        'loss': 'focal',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },
    {
        'name': 'loss_tversky',
        'loss': 'tversky',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },

    # ========================================================================
    # Experiment 2: Learning Rate Tuning
    # ========================================================================
    {
        'name': 'lr_1e3',
        'loss': 'bce_dice',
        'lr': 1e-3,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },
    {
        'name': 'lr_5e4',
        'loss': 'bce_dice',
        'lr': 5e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },
    {
        'name': 'lr_5e5',
        'loss': 'bce_dice',
        'lr': 5e-5,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },

    # ========================================================================
    # Experiment 3: Batch Size Impact
    # ========================================================================
    {
        'name': 'batch_2',
        'loss': 'bce_dice',
        'lr': 1e-4,
        'batch_size': 2,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },
    {
        'name': 'batch_8',
        'loss': 'bce_dice',
        'lr': 1e-4,
        'batch_size': 8,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },

    # ========================================================================
    # Experiment 4: Optimizer Comparison
    # ========================================================================
    {
        'name': 'opt_adam',
        'loss': 'bce_dice',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },
    {
        'name': 'opt_sgd',
        'loss': 'bce_dice',
        'lr': 1e-3,  # SGD typically needs higher LR
        'batch_size': 4,
        'optimizer': 'sgd',
        'scheduler': 'cosine',
        'augmentation': 'train',
    },

    # ========================================================================
    # Experiment 5: Scheduler Comparison
    # ========================================================================
    {
        'name': 'sched_step',
        'loss': 'bce_dice',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'step',
        'augmentation': 'train',
    },
    {
        'name': 'sched_plateau',
        'loss': 'bce_dice',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'plateau',
        'augmentation': 'train',
    },

    # ========================================================================
    # Experiment 6: Augmentation Impact
    # ========================================================================
    {
        'name': 'aug_none',
        'loss': 'bce_dice',
        'lr': 1e-4,
        'batch_size': 4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'augmentation': 'val',  # val augmentation = minimal (just resize)
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_loss_function(loss_name):
    """Get loss function by name."""
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'dice':
        return smp.losses.DiceLoss(mode='binary')
    elif loss_name == 'bce_dice':
        bce = nn.BCEWithLogitsLoss()
        dice = smp.losses.DiceLoss(mode='binary')
        return lambda pred, target: bce(pred, target) + dice(pred, target)
    elif loss_name == 'focal':
        return smp.losses.FocalLoss(mode='binary')
    elif loss_name == 'tversky':
        return smp.losses.TverskyLoss(mode='binary', alpha=0.7, beta=0.3)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    iou_scores = AverageMeter()
    dice_scores = AverageMeter()
    image_acc_scores = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            metrics = calculate_metrics(preds, masks)
            img_metrics = calculate_image_level_accuracy(preds, masks)

        losses.update(loss.item(), images.size(0))
        iou_scores.update(metrics['iou'], images.size(0))
        dice_scores.update(metrics['dice'], images.size(0))
        image_acc_scores.update(img_metrics['image_accuracy'], images.size(0))

        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'iou': f'{iou_scores.avg:.4f}'})

    return {
        'loss': losses.avg,
        'iou': iou_scores.avg,
        'dice': dice_scores.avg,
        'image_accuracy': image_acc_scores.avg
    }


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    losses = AverageMeter()
    iou_scores = AverageMeter()
    dice_scores = AverageMeter()
    image_acc_scores = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.sigmoid(outputs) > 0.5
            metrics = calculate_metrics(preds, masks)
            img_metrics = calculate_image_level_accuracy(preds, masks)

            losses.update(loss.item(), images.size(0))
            iou_scores.update(metrics['iou'], images.size(0))
            dice_scores.update(metrics['dice'], images.size(0))
            image_acc_scores.update(img_metrics['image_accuracy'], images.size(0))

            pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'iou': f'{iou_scores.avg:.4f}'})

    return {
        'loss': losses.avg,
        'iou': iou_scores.avg,
        'dice': dice_scores.avg,
        'image_accuracy': image_acc_scores.avg
    }


def plot_model_comparison(all_histories, output_dir):
    """Plot comparison of all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Training IoU
    for exp_name, history in all_histories.items():
        axes[0, 0].plot(history['train_iou'], label=exp_name, linewidth=2)
    axes[0, 0].set_title('Training IoU Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('IoU')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Validation IoU
    for exp_name, history in all_histories.items():
        axes[0, 1].plot(history['val_iou'], label=exp_name, linewidth=2)
    axes[0, 1].set_title('Validation IoU Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Validation Loss
    for exp_name, history in all_histories.items():
        axes[1, 0].plot(history['val_loss'], label=exp_name, linewidth=2)
    axes[1, 0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Image-Level Accuracy
    for exp_name, history in all_histories.items():
        axes[1, 1].plot(history['val_image_accuracy'], label=exp_name, linewidth=2)
    axes[1, 1].set_title('Validation Image Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Image Accuracy')
    axes[1, 1].legend(loc='best', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'stage4_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_experiment(config, device, train_loader, val_loader):
    """Train a single experiment configuration."""

    print(f"\n{'='*70}")
    print(f"Experiment: {config['name']}")
    print(f"{'='*70}")
    print(f"Loss: {config['loss']}")
    print(f"LR: {config['lr']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Scheduler: {config['scheduler']}")
    print(f"Augmentation: {config['augmentation']}")
    print()

    # Create model
    model = BASE_MODEL_FN()
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    # Loss function
    criterion = get_loss_function(config['loss'])

    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)

    # Scheduler
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS//3, gamma=0.1)
    elif config['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    else:
        scheduler = None

    # Training loop
    best_iou = 0.0
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [], 'train_image_accuracy': [],
        'val_loss': [], 'val_iou': [], 'val_dice': [], 'val_image_accuracy': [], 'lr': []
    }

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)

        lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if config['scheduler'] == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        history['train_loss'].append(train_metrics['loss'])
        history['train_iou'].append(train_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['train_image_accuracy'].append(train_metrics['image_accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_image_accuracy'].append(val_metrics['image_accuracy'])
        history['lr'].append(lr)

        print(f"\nEpoch {epoch}/{EPOCHS}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"  LR: {lr:.6f}")

        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']

        # Clear GPU cache
        torch.cuda.empty_cache()

    print(f"\n{config['name']} - Best IoU: {best_iou:.4f}\n")

    # Clean up
    del model, optimizer
    if scheduler:
        del scheduler
    torch.cuda.empty_cache()

    return {
        'name': config['name'],
        'loss': config['loss'],
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'optimizer': config['optimizer'],
        'scheduler': config['scheduler'],
        'augmentation': config['augmentation'],
        'best_iou': best_iou,
        'final_val_loss': history['val_loss'][-1],
        'final_val_iou': history['val_iou'][-1],
        'history': history,
    }


def main():
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"STAGE 4: HYPERPARAMETER TUNING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)

    # Load datasets
    train_dataset = ForgeryDataset(
        f"{DATA_DIR}/train/images",
        f"{DATA_DIR}/train/masks",
        augmentation=get_augmentation('train')
    )
    val_dataset = ForgeryDataset(
        f"{DATA_DIR}/val/images",
        f"{DATA_DIR}/val/masks",
        augmentation=get_augmentation('val')
    )

    # Limit samples if specified
    if MAX_SAMPLES and MAX_SAMPLES < len(train_dataset):
        train_indices = np.random.choice(len(train_dataset), MAX_SAMPLES, replace=False)
        val_indices = np.random.choice(len(val_dataset), int(MAX_SAMPLES*0.2), replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Load existing results if available
    results = []
    all_histories = {}

    intermediate_csv = output_dir / 'stage4_leaderboard_intermediate.csv'
    intermediate_pkl = output_dir / 'stage4_histories_intermediate.pkl'

    if intermediate_csv.exists():
        existing_results = pd.read_csv(intermediate_csv).to_dict('records')
        results.extend(existing_results)
        print(f"📂 Loaded {len(existing_results)} existing results from intermediate file")

    if intermediate_pkl.exists():
        with open(intermediate_pkl, 'rb') as f:
            all_histories = pickle.load(f)
        print(f"📂 Loaded {len(all_histories)} existing training histories\n")

    # Run experiments
    for i, config in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] Starting: {config['name']}")

        # Create dataloaders with experiment-specific batch size and augmentation
        train_aug = get_augmentation(config['augmentation'])
        train_dataset_exp = ForgeryDataset(
            f"{DATA_DIR}/train/images",
            f"{DATA_DIR}/train/masks",
            augmentation=train_aug
        )
        if MAX_SAMPLES and MAX_SAMPLES < len(train_dataset_exp):
            train_dataset_exp = Subset(train_dataset_exp, train_indices)

        train_loader = DataLoader(
            train_dataset_exp,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        # Train experiment
        result = train_experiment(config, device, train_loader, val_loader)
        results.append(result)

        # Store history
        all_histories[config['name']] = pd.DataFrame(result['history'])

        # Save intermediate results
        pd.DataFrame(results).to_csv(intermediate_csv, index=False)
        with open(intermediate_pkl, 'wb') as f:
            pickle.dump(all_histories, f)

        print(f"✓ Intermediate results saved (Total: {len(results)} experiments complete)")

    # Print final results
    print(f"\n{'='*70}")
    print(f"STAGE 4 FINAL RESULTS - LEADERBOARD")
    print(f"{'='*70}\n")

    # Sort by best IoU
    results_sorted = sorted(results, key=lambda x: x['best_iou'], reverse=True)

    # Print table
    print(f"{'Rank':<6}{'Experiment':<25}{'Best IoU':<12}{'Loss Fn':<12}{'LR':<10}")
    print(f"{'-'*70}")
    for i, result in enumerate(results_sorted, 1):
        print(f"{i:<6}{result['name']:<25}{result['best_iou']:<12.4f}{result['loss']:<12}{result['lr']:<10.6f}")

    # Save final leaderboard
    pd.DataFrame(results_sorted).to_csv(output_dir / 'stage4_leaderboard.csv', index=False)
    print(f"\nLeaderboard saved to: experiments/stage4_leaderboard.csv")

    # Plot comparison
    if all_histories:
        plot_model_comparison(all_histories, output_dir)
        print(f"Comparison graph saved to: experiments/stage4_comparison.png")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
