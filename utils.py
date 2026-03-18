"""
Utility functions for training and evaluation.
"""

import torch
import numpy as np
from pathlib import Path
import shutil


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate segmentation metrics.

    Args:
        pred: Predicted masks (B, 1, H, W) - binary or probability
        target: Ground truth masks (B, 1, H, W) - binary
        threshold: Threshold for converting probabilities to binary

    Returns:
        Dictionary with metrics
    """
    # Ensure binary predictions and convert to float
    if pred.dtype == torch.bool:
        pred = pred.float()
    elif pred.dtype == torch.float32:
        pred = (pred > threshold).float()

    # Ensure target is float
    target = target.float()

    pred = pred.view(-1)
    target = target.view(-1)

    # True Positives, False Positives, False Negatives, True Negatives
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    # IoU (Intersection over Union)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)

    # Dice coefficient (F1 score)
    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)

    # Pixel accuracy
    accuracy = (tp + tn + 1e-7) / (tp + fp + fn + tn + 1e-7)

    # Precision
    precision = (tp + 1e-7) / (tp + fp + 1e-7)

    # Recall
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    # F1 score (same as Dice for binary segmentation)
    f1 = (2 * precision * recall + 1e-7) / (precision + recall + 1e-7)

    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def calculate_image_level_accuracy(pred, target):
    """
    Calculate image-level classification accuracy.

    Classifies entire images as forged/authentic:
    - If predicted mask has ANY forged pixels (>0) → classified as FORGED
    - If predicted mask has NO forged pixels (all 0) → classified as AUTHENTIC

    Args:
        pred: Predicted masks (B, 1, H, W) - binary (0 or 1)
        target: Ground truth masks (B, 1, H, W) - binary (0 or 1)

    Returns:
        Dictionary with image-level metrics
    """
    batch_size = pred.shape[0]

    # For each image in batch, check if it has ANY forged pixels
    # Sum over spatial dimensions (H, W) to get per-image forgery count
    pred_has_forgery = (pred.view(batch_size, -1).sum(dim=1) > 0).float()  # (B,)
    target_has_forgery = (target.view(batch_size, -1).sum(dim=1) > 0).float()  # (B,)

    # Calculate image-level accuracy
    correct = (pred_has_forgery == target_has_forgery).float().sum()
    image_accuracy = correct / batch_size

    # Calculate image-level metrics
    # True Positives: predicted forged AND actually forged
    tp_img = ((pred_has_forgery == 1) & (target_has_forgery == 1)).float().sum()

    # False Positives: predicted forged BUT actually authentic
    fp_img = ((pred_has_forgery == 1) & (target_has_forgery == 0)).float().sum()

    # False Negatives: predicted authentic BUT actually forged
    fn_img = ((pred_has_forgery == 0) & (target_has_forgery == 1)).float().sum()

    # True Negatives: predicted authentic AND actually authentic
    tn_img = ((pred_has_forgery == 0) & (target_has_forgery == 0)).float().sum()

    # Image-level precision, recall, F1
    img_precision = (tp_img + 1e-7) / (tp_img + fp_img + 1e-7)
    img_recall = (tp_img + 1e-7) / (tp_img + fn_img + 1e-7)
    img_f1 = (2 * img_precision * img_recall + 1e-7) / (img_precision + img_recall + 1e-7)

    return {
        'image_accuracy': image_accuracy.item(),
        'image_precision': img_precision.item(),
        'image_recall': img_recall.item(),
        'image_f1': img_f1.item(),
        'tp_images': tp_img.item(),
        'fp_images': fp_img.item(),
        'fn_images': fn_img.item(),
        'tn_images': tn_img.item()
    }


def save_checkpoint(state, is_best, output_dir):
    """
    Save model checkpoint.

    Args:
        state: Dictionary with model state, optimizer state, etc.
        is_best: Boolean indicating if this is the best model so far
        output_dir: Directory to save checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save last checkpoint
    checkpoint_path = output_dir / 'checkpoint_last.pth'
    torch.save(state, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = output_dir / 'checkpoint_best.pth'
        shutil.copyfile(checkpoint_path, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint


def count_parameters(model):
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total and trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable
    }


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
