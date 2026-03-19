"""
Simple Training Script - Just edit the variables below and run!

Usage: python train_simple.py
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

from Models.deeplabv3plus import get_deeplabv3plus
from Models.unet import PretrainedHybridUNet, DoubleConv, ResBlock
from Models.fpn import (
    create_fpn_full, create_fpn_no_lateral, create_fpn_no_topdown,
    create_fpn_shallow, create_fpn_deep_decoder, create_fpn_minimal
)

# ============================================================================
# MULTI-MODEL TRAINING MODE
# ============================================================================
# Set to True to train multiple models automatically
# Set to False to train just one model (defined below)
MULTI_MODEL_MODE = False

# If MULTI_MODEL_MODE = True, define models to compare:
MODELS_TO_COMPARE = [
    # ========================================================================
    # STAGE 1: Architecture Comparison (All ResNet34) - COMPLETED
    # Winner: FPN (0.2141 IoU)
    # ========================================================================
    # {'name': 'smp_unet', 'model': lambda: smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'smp_unetplusplus', 'model': lambda: smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'smp_deeplabv3plus', 'model': lambda: smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'smp_fpn', 'model': lambda: smp.FPN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'smp_pspnet', 'model': lambda: smp.PSPNet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'smp_manet', 'model': lambda: smp.MAnet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'smp_linknet', 'model': lambda: smp.Linknet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'smp_pan', 'model': lambda: smp.PAN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},

    # ========================================================================
    # STAGE 2: Backbone Comparison for FPN (Winner from Stage 1) - COMPLETED
    # Winner: FPN + ResNeXt50 (0.2659 IoU)
    # ========================================================================
    # {'name': 'fpn_resnet34', 'model': lambda: smp.FPN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'fpn_resnet50', 'model': lambda: smp.FPN(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'fpn_se_resnet50', 'model': lambda: smp.FPN(encoder_name='se_resnet50', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'fpn_efficientnet_b4', 'model': lambda: smp.FPN(encoder_name='efficientnet-b4', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'fpn_efficientnet_b5', 'model': lambda: smp.FPN(encoder_name='efficientnet-b5', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},
    # {'name': 'fpn_resnext50', 'model': lambda: smp.FPN(encoder_name='resnext50_32x4d', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)},

    # ========================================================================
    # STAGE 3: FPN Architecture Refinement (Ablation Studies) using ResNeXt50 backbone
    # ========================================================================

    # # Baseline - Full FPN (all components enabled)
    # {'name': 'fpn_full', 'model': lambda: create_fpn_full(backbone_name='resnext50_32x4d')},

    # # Ablation 1: Remove lateral connections (tests skip connection importance)
    # {'name': 'fpn_no_lateral', 'model': lambda: create_fpn_no_lateral(backbone_name='resnext50_32x4d')},

    # # Ablation 2: Remove top-down pathway (no multi-scale fusion)
    # {'name': 'fpn_no_topdown', 'model': lambda: create_fpn_no_topdown(backbone_name='resnext50_32x4d')},

    # # Ablation 3: Shallow pyramid (2 levels instead of 4 - tests multi-scale importance)
    # {'name': 'fpn_shallow', 'model': lambda: create_fpn_shallow(backbone_name='resnext50_32x4d')},

    # # Ablation 4: Minimal FPN (remove everything - worst case baseline)
    # {'name': 'fpn_minimal', 'model': lambda: create_fpn_minimal(backbone_name='resnext50_32x4d')},
]

# ============================================================================
# SINGLE MODEL MODE (when MULTI_MODEL_MODE = False)
# ============================================================================
EXPERIMENT_NAME = "final_model_optimized"

# Data
DATA_DIR = "Datasets/unified"
MAX_SAMPLES = None  # Use full dataset for final model training

# Final optimized model from Stages 1-4:
# - Stage 1 winner: FPN architecture
# - Stage 2 winner: ResNeXt50 backbone
# - Stage 3 insight: Top-down pathway is critical
# - Stage 4 optimization: batch_size=2, optimizer=adam
MODEL = smp.FPN(encoder_name='resnext50_32x4d', encoder_weights='imagenet', in_channels=3, classes=1, activation=None)

# Training settings - OPTIMIZED FROM STAGE 4
EPOCHS = 30  # More epochs for full dataset training
BATCH_SIZE = 2  # OPTIMIZED: Reduced from 4 (+19.7% improvement)
LEARNING_RATE = 1e-4  # OPTIMIZED: Already optimal from baseline
OPTIMIZER = "adam"  # OPTIMIZED: Changed from adamw (+2.9% improvement)
SCHEDULER = "cosine"  # OPTIMIZED: Already optimal from baseline
LOSS = "bce_dice"  # OPTIMIZED: Already optimal from baseline
AUGMENTATION = True  # OPTIMIZED: Already optimal from baseline

# System
NUM_WORKERS = 4
SEED = 42

def get_loss_function(loss_name):
    """Get loss function."""
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

        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'iou': f'{iou_scores.avg:.4f}', 'img_acc': f'{image_acc_scores.avg:.4f}'})

    return {'loss': losses.avg, 'iou': iou_scores.avg, 'dice': dice_scores.avg, 'image_accuracy': image_acc_scores.avg}


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

            pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'iou': f'{iou_scores.avg:.4f}', 'img_acc': f'{image_acc_scores.avg:.4f}'})

    return {'loss': losses.avg, 'iou': iou_scores.avg, 'dice': dice_scores.avg, 'image_accuracy': image_acc_scores.avg}


def plot_model_comparison(all_histories, output_dir):
    """Plot comparison of all models' IoU over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Training IoU
    for model_name, history in all_histories.items():
        axes[0, 0].plot(history['train_iou'], label=model_name, linewidth=2)
    axes[0, 0].set_title('Training IoU Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('IoU')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Validation IoU
    for model_name, history in all_histories.items():
        axes[0, 1].plot(history['val_iou'], label=model_name, linewidth=2)
    axes[0, 1].set_title('Validation IoU Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Validation Loss
    for model_name, history in all_histories.items():
        axes[1, 0].plot(history['val_loss'], label=model_name, linewidth=2)
    axes[1, 0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Image-Level Accuracy
    for model_name, history in all_histories.items():
        axes[1, 1].plot(history['val_image_accuracy'], label=model_name, linewidth=2)
    axes[1, 1].set_title('Validation Image Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Image Accuracy')
    axes[1, 1].legend(loc='best', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_curves(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_iou'], label='Train')
    axes[0, 1].plot(history['val_iou'], label='Val')
    axes[0, 1].set_title('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(history['train_dice'], label='Train')
    axes[0, 2].plot(history['val_dice'], label='Val')
    axes[0, 2].set_title('Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 0].plot(history['train_image_accuracy'], label='Train')
    axes[1, 0].plot(history['val_image_accuracy'], label='Val')
    axes[1, 0].set_title('Image-Level Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].grid(True)

    # Hide the last subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'curves.png', dpi=150)
    plt.close()


def train_single_model(model, model_name, device, train_loader, val_loader):
    """Train a single model and return results (no individual model folders saved)."""

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}\n")

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB\n")

    # Loss, optimizer, scheduler
    criterion = get_loss_function(LOSS)

    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    elif OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

    if SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS//3, gamma=0.1)
    elif SCHEDULER == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    else:
        scheduler = None

    # Training loop
    best_iou = 0.0
    history = {'train_loss': [], 'train_iou': [], 'train_dice': [], 'train_image_accuracy': [],
               'val_loss': [], 'val_iou': [], 'val_dice': [], 'val_image_accuracy': [], 'lr': []}

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)

        lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if SCHEDULER == 'plateau':
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
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}, Img Acc: {train_metrics['image_accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}, Img Acc: {val_metrics['image_accuracy']:.4f}")
        print(f"  LR: {lr:.6f}")

        # Debug: Check if model is predicting anything
        if epoch == 1:
            print(f"  [DEBUG] First epoch - check if metrics are non-zero above")

        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']

        # Clear GPU cache after each epoch to prevent memory buildup
        torch.cuda.empty_cache()

    print(f"\n{model_name} - Best IoU: {best_iou:.4f}\n")

    # Clean up
    del model, optimizer
    if scheduler:
        del scheduler
    torch.cuda.empty_cache()

    return {
        'name': model_name,
        'best_iou': best_iou,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_iou': history['train_iou'][-1],
        'final_val_iou': history['val_iou'][-1],
        'history': history,  # Keep history for comparison graph
    }


def main():
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("WARNING: Training on CPU!\n")

    # Datasets
    train_dataset = ForgeryDataset(
        f"{DATA_DIR}/train/images",
        f"{DATA_DIR}/train/masks",
        augmentation=get_augmentation('train') if AUGMENTATION else None
    )
    val_dataset = ForgeryDataset(
        f"{DATA_DIR}/val/images",
        f"{DATA_DIR}/val/masks",
        augmentation=get_augmentation('val')
    )

    # Limit samples
    if MAX_SAMPLES and MAX_SAMPLES < len(train_dataset):
        train_dataset = Subset(train_dataset, np.random.choice(len(train_dataset), MAX_SAMPLES, replace=False))
        val_dataset = Subset(val_dataset, np.random.choice(len(val_dataset), int(MAX_SAMPLES*0.2), replace=False))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Choose mode
    if MULTI_MODEL_MODE:
        # Train multiple models for comparison
        print(f"\n{'='*70}")
        print(f"MULTI-MODEL COMPARISON MODE")
        print(f"Training {len(MODELS_TO_COMPARE)} models with {EPOCHS} epochs each")
        print(f"{'='*70}\n")

        results = []
        all_histories = {}  # Store histories for comparison graph

        # Load existing intermediate results if they exist (to resume training)
        intermediate_csv = Path('experiments/leaderboard_intermediate.csv')
        intermediate_pkl = Path('experiments/histories_intermediate.pkl')

        if intermediate_csv.exists():
            existing_results = pd.read_csv(intermediate_csv).to_dict('records')
            results.extend(existing_results)
            print(f"📂 Loaded {len(existing_results)} existing results from intermediate file")

        if intermediate_pkl.exists():
            with open(intermediate_pkl, 'rb') as f:
                all_histories = pickle.load(f)
            print(f"📂 Loaded {len(all_histories)} existing training histories")

        for i, model_config in enumerate(MODELS_TO_COMPARE):
            print(f"\n[{i+1}/{len(MODELS_TO_COMPARE)}] Starting: {model_config['name']}")
            model = model_config['model']()  # Create model instance
            result = train_single_model(model, model_config['name'], device, train_loader, val_loader)
            results.append(result)

            # Store history for comparison graph
            all_histories[model_config['name']] = pd.DataFrame(result['history'])

            # Save intermediate results after each model (in case of crash)
            pd.DataFrame(results).to_csv('experiments/leaderboard_intermediate.csv', index=False)

            # Save intermediate histories (for plotting if crash occurs)
            with open('experiments/histories_intermediate.pkl', 'wb') as f:
                pickle.dump(all_histories, f)

            print(f"✓ Intermediate results saved (Total: {len(results)} models complete)")

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS - LEADERBOARD")
        print(f"{'='*70}\n")

        # Sort by best IoU
        results = sorted(results, key=lambda x: x['best_iou'], reverse=True)

        # Print table
        print(f"{'Rank':<6}{'Model':<30}{'Best IoU':<12}{'Val Loss':<12}")
        print(f"{'-'*70}")
        for i, result in enumerate(results, 1):
            print(f"{i:<6}{result['name']:<30}{result['best_iou']:<12.4f}{result['final_val_loss']:<12.4f}")

        # Save leaderboard
        pd.DataFrame(results).to_csv('experiments/leaderboard.csv', index=False)
        print(f"\nLeaderboard saved to: experiments/leaderboard.csv")

        # Plot comparison graph
        if all_histories:
            plot_model_comparison(all_histories, Path('experiments'))
            print(f"Model comparison graph saved to: experiments/model_comparison.png")

        print(f"{'='*70}\n")

    else:
        # Train single model
        result = train_single_model(MODEL, EXPERIMENT_NAME, device, train_loader, val_loader)

        print(f"\n{'='*70}")
        print(f"Training complete! Best IoU: {result['best_iou']:.4f}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
