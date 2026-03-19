"""
Test Final Model on Test Set

Evaluates the trained model and visualizes predictions vs ground truth.

Usage: python test_model.py
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from dataset import ForgeryDataset, get_augmentation
from utils import calculate_metrics, calculate_image_level_accuracy, AverageMeter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model checkpoint to test
MODEL_CHECKPOINT = "experiments/final_model_optimized/best_model.pth"

# Data
DATA_DIR = "Datasets/unified"
BATCH_SIZE = 4
NUM_WORKERS = 4

# Number of examples to visualize
NUM_VISUALIZATIONS = 10

# Number of best/worst examples for analysis
NUM_BEST = 2
NUM_WORST = 2
NUM_TRUE_NEGATIVES = 0
NUM_FALSE_NEGATIVES = 0

# Output directory
OUTPUT_DIR = Path("experiments/final_model_optimized/test_results")


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def test_model(model, dataloader, device):
    """Evaluate model on test set and collect detailed per-sample results."""
    model.eval()

    iou_scores = AverageMeter()
    dice_scores = AverageMeter()
    pixel_acc_scores = AverageMeter()
    image_acc_scores = AverageMeter()
    precision_scores = AverageMeter()
    recall_scores = AverageMeter()
    f1_scores = AverageMeter()

    all_predictions = []
    all_targets = []
    all_images = []

    # Collect per-sample results for analysis
    sample_results = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            # Calculate metrics
            metrics = calculate_metrics(preds, masks)
            img_metrics = calculate_image_level_accuracy(preds, masks)

            iou_scores.update(metrics['iou'], images.size(0))
            dice_scores.update(metrics['dice'], images.size(0))
            pixel_acc_scores.update(metrics['accuracy'], images.size(0))
            image_acc_scores.update(img_metrics['image_accuracy'], images.size(0))
            precision_scores.update(metrics['precision'], images.size(0))
            recall_scores.update(metrics['recall'], images.size(0))
            f1_scores.update(metrics['f1'], images.size(0))

            # Store for visualization
            all_predictions.append(preds.cpu())
            all_targets.append(masks.cpu())
            all_images.append(images.cpu())

            # Collect per-sample results for best/worst analysis
            for i in range(images.size(0)):
                img = images[i].cpu()
                pred = preds[i].cpu().squeeze()
                target = masks[i].cpu().squeeze()

                # Calculate IoU for this specific sample
                pred_np = pred.numpy().astype(bool)
                target_np = target.numpy().astype(bool)

                intersection = np.logical_and(pred_np, target_np).sum()
                union = np.logical_or(pred_np, target_np).sum()
                iou = intersection / (union + 1e-7)

                # Check if image has forgery
                has_forgery = target_np.any()
                predicted_forgery = pred_np.any()

                # Categorize
                if has_forgery:
                    if predicted_forgery:
                        category = "detected_forgery"
                    else:
                        category = "missed_forgery"  # False Negative
                else:
                    if predicted_forgery:
                        category = "false_alarm"  # False Positive
                    else:
                        category = "correct_authentic"  # True Negative

                sample_results.append({
                    'image': img,
                    'prediction': pred,
                    'target': target,
                    'iou': iou,
                    'has_forgery': has_forgery,
                    'predicted_forgery': predicted_forgery,
                    'category': category,
                })

    results = {
        'IoU': iou_scores.avg,
        'Dice': dice_scores.avg,
        'Pixel Accuracy': pixel_acc_scores.avg,
        'Image Accuracy': image_acc_scores.avg,
        'Precision': precision_scores.avg,
        'Recall': recall_scores.avg,
        'F1 Score': f1_scores.avg,
    }

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_images = torch.cat(all_images, dim=0)

    return results, all_images, all_predictions, all_targets, sample_results


def visualize_predictions(images, predictions, targets, num_samples, output_dir):
    """Visualize model predictions vs ground truth."""

    # Denormalize images (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Select random samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)

    # Create visualization grid
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        img = images[idx]
        pred = predictions[idx].squeeze()
        target = targets[idx].squeeze()

        # Denormalize image
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Convert masks to numpy
        pred = pred.numpy()
        target = target.numpy()

        # Plot
        if num_samples == 1:
            ax_img, ax_gt, ax_pred = axes[0], axes[1], axes[2]
        else:
            ax_img, ax_gt, ax_pred = axes[i, 0], axes[i, 1], axes[i, 2]

        # Original image
        ax_img.imshow(img)
        ax_img.set_title(f'Sample {idx}: Original Image')
        ax_img.axis('off')

        # Ground truth
        ax_gt.imshow(img)
        ax_gt.imshow(target, alpha=0.5, cmap='Reds')
        ax_gt.set_title('Ground Truth (Red = Forged)')
        ax_gt.axis('off')

        # Prediction
        ax_pred.imshow(img)
        ax_pred.imshow(pred, alpha=0.5, cmap='Blues')
        ax_pred.set_title('Prediction (Blue = Forged)')
        ax_pred.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to: {output_dir / 'predictions_visualization.png'}")


def visualize_comparison_grid(images, predictions, targets, num_samples, output_dir):
    """Create a detailed comparison grid with overlay visualization."""

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Select random samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)

    # Create visualization grid (4 columns: image, GT, pred, overlay)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i, idx in enumerate(indices):
        img = images[idx]
        pred = predictions[idx].squeeze()
        target = targets[idx].squeeze()

        # Denormalize image
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Convert masks to numpy
        pred = pred.numpy()
        target = target.numpy()

        # Calculate IoU for this sample
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        iou = intersection / (union + 1e-7)

        # Plot
        if num_samples == 1:
            ax_img, ax_gt, ax_pred, ax_overlay = axes[0], axes[1], axes[2], axes[3]
        else:
            ax_img, ax_gt, ax_pred, ax_overlay = axes[i, 0], axes[i, 1], axes[i, 2], axes[i, 3]

        # Original image
        ax_img.imshow(img)
        ax_img.set_title(f'Original Image')
        ax_img.axis('off')

        # Ground truth mask only
        ax_gt.imshow(target, cmap='gray', vmin=0, vmax=1)
        ax_gt.set_title('Ground Truth Mask')
        ax_gt.axis('off')

        # Prediction mask only
        ax_pred.imshow(pred, cmap='gray', vmin=0, vmax=1)
        ax_pred.set_title(f'Predicted Mask (IoU: {iou:.3f})')
        ax_pred.axis('off')

        # Overlay: Green = TP, Red = FN, Blue = FP, Black = TN
        # Convert to boolean for logical operations
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)

        overlay = np.zeros((*pred.shape, 3))
        overlay[np.logical_and(pred_bool, target_bool)] = [0, 1, 0]  # TP: Green
        overlay[np.logical_and(~pred_bool, target_bool)] = [1, 0, 0]  # FN: Red
        overlay[np.logical_and(pred_bool, ~target_bool)] = [0, 0, 1]  # FP: Blue

        ax_overlay.imshow(img)
        ax_overlay.imshow(overlay, alpha=0.5)
        ax_overlay.set_title('TP:Green | FN:Red | FP:Blue')
        ax_overlay.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Detailed comparison saved to: {output_dir / 'detailed_comparison.png'}")


def visualize_analysis_samples(samples, title, output_path):
    """Visualize a set of samples in a grid for best/worst analysis."""
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, sample in enumerate(samples):
        img = sample['image']
        pred = sample['prediction'].numpy()
        target = sample['target'].numpy()
        iou = sample['iou']

        # Denormalize image
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image\nIoU: {iou:.3f}')
        axes[i, 0].axis('off')

        # Ground truth
        axes[i, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')

        # Prediction
        axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

        # Overlay with TP/FP/FN
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)

        overlay = np.zeros((*pred.shape, 3))
        overlay[np.logical_and(pred_bool, target_bool)] = [0, 1, 0]  # TP: Green
        overlay[np.logical_and(~pred_bool, target_bool)] = [1, 0, 0]  # FN: Red
        overlay[np.logical_and(pred_bool, ~target_bool)] = [0, 0, 1]  # FP: Blue

        axes[i, 3].imshow(img)
        axes[i, 3].imshow(overlay, alpha=0.6)
        axes[i, 3].set_title('Overlay\nGreen:TP | Red:FN | Blue:FP')
        axes[i, 3].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_best_worst(sample_results, output_dir):
    """Analyze and visualize best/worst predictions."""
    print(f"\n{'='*70}")
    print(f"ANALYZING BEST AND WORST PREDICTIONS")
    print(f"{'='*70}\n")

    # Separate by category
    forged_detected = [r for r in sample_results if r['category'] == 'detected_forgery']
    missed_forgeries = [r for r in sample_results if r['category'] == 'missed_forgery']
    correct_authentic = [r for r in sample_results if r['category'] == 'correct_authentic']
    false_alarms = [r for r in sample_results if r['category'] == 'false_alarm']

    print(f"Category Breakdown:")
    print(f"  Forged images with detection: {len(forged_detected)}")
    print(f"  Missed forgeries (FN): {len(missed_forgeries)}")
    print(f"  Correctly identified authentic (TN): {len(correct_authentic)}")
    print(f"  False alarms (FP): {len(false_alarms)}\n")

    # Sort forged detections by IoU
    forged_detected_sorted = sorted(forged_detected, key=lambda x: x['iou'], reverse=True)

    # Get best predictions (highest IoU on forged images)
    best_samples = forged_detected_sorted[:NUM_BEST] if forged_detected_sorted else []
    if best_samples:
        print(f"Best predictions (highest IoU on forged images):")
        for i, s in enumerate(best_samples, 1):
            print(f"  {i}. IoU: {s['iou']:.4f}")

    # Get worst predictions (lowest IoU on forged images)
    worst_samples = forged_detected_sorted[-NUM_WORST:][::-1] if len(forged_detected_sorted) >= NUM_WORST else []
    if worst_samples:
        print(f"\nWorst predictions (lowest IoU on detected forged images):")
        for i, s in enumerate(worst_samples, 1):
            print(f"  {i}. IoU: {s['iou']:.4f}")

    # Get missed forgeries
    missed_samples = missed_forgeries[:NUM_FALSE_NEGATIVES] if missed_forgeries else []
    if missed_samples:
        print(f"\nMissed forgeries (False Negatives): {len(missed_samples)} examples")

    # Get correct authentic
    tn_samples = correct_authentic[:NUM_TRUE_NEGATIVES] if correct_authentic else []
    if tn_samples:
        print(f"\nCorrect authentic identifications (True Negatives): {len(tn_samples)} examples")

    print()

    # Visualize each category
    if best_samples:
        visualize_analysis_samples(
            best_samples,
            f"BEST PREDICTIONS (Top {len(best_samples)} - Highest IoU on Forged Images)",
            output_dir / "best_predictions.png"
        )
        print(f"✓ Saved: {output_dir / 'best_predictions.png'}")

    if worst_samples:
        visualize_analysis_samples(
            worst_samples,
            f"WORST PREDICTIONS (Bottom {len(worst_samples)} - Lowest IoU on Detected Forgeries)",
            output_dir / "worst_predictions.png"
        )
        print(f"✓ Saved: {output_dir / 'worst_predictions.png'}")

    if missed_samples:
        visualize_analysis_samples(
            missed_samples,
            f"MISSED FORGERIES (False Negatives - Model Failed to Detect)",
            output_dir / "missed_forgeries.png"
        )
        print(f"✓ Saved: {output_dir / 'missed_forgeries.png'}")

    if tn_samples:
        visualize_analysis_samples(
            tn_samples,
            f"CORRECT AUTHENTIC (True Negatives - Correctly Identified as Authentic)",
            output_dir / "true_negatives.png"
        )
        print(f"✓ Saved: {output_dir / 'true_negatives.png'}")

    # Create summary statistics
    summary_path = output_dir / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL PERFORMANCE ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total test samples: {len(sample_results)}\n\n")

        f.write("Category Breakdown:\n")
        f.write(f"  Forged images with detection: {len(forged_detected)}\n")
        if forged_detected_sorted:
            f.write(f"    - Best IoU: {forged_detected_sorted[0]['iou']:.4f}\n")
            f.write(f"    - Worst IoU: {forged_detected_sorted[-1]['iou']:.4f}\n")
            f.write(f"    - Mean IoU: {np.mean([r['iou'] for r in forged_detected]):.4f}\n")
        f.write(f"\n")
        f.write(f"  Missed forgeries (FN): {len(missed_forgeries)}\n")
        f.write(f"  False alarms (FP): {len(false_alarms)}\n")
        f.write(f"  Correct authentic (TN): {len(correct_authentic)}\n")
        f.write("\n")

        # Calculate rates
        total_forged = len(forged_detected) + len(missed_forgeries)
        total_authentic = len(correct_authentic) + len(false_alarms)

        if total_forged > 0:
            detection_rate = len(forged_detected) / total_forged * 100
            f.write(f"Detection Rate: {detection_rate:.1f}% ({len(forged_detected)}/{total_forged})\n")

        if total_authentic > 0:
            tn_rate = len(correct_authentic) / total_authentic * 100
            f.write(f"True Negative Rate: {tn_rate:.1f}% ({len(correct_authentic)}/{total_authentic})\n")

        f.write("\n" + "="*70 + "\n")

    print(f"✓ Saved: {summary_path}")

    print(f"{'='*70}\n")


def save_metrics_report(results, output_dir, checkpoint_path):
    """Save test metrics to text file and CSV."""

    # Text report
    report_path = output_dir / 'test_metrics.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FINAL MODEL TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {checkpoint_path}\n\n")
        f.write("Segmentation Metrics:\n")
        f.write("-"*70 + "\n")
        for metric, value in results.items():
            f.write(f"{metric:<20}: {value:.4f}\n")
        f.write("="*70 + "\n")

    print(f"✓ Test report saved to: {report_path}")

    # CSV for easy import
    csv_path = output_dir / 'test_metrics.csv'
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    print(f"✓ Test metrics CSV saved to: {csv_path}")


def main():
    print(f"\n{'='*70}")
    print(f"TESTING FINAL MODEL")
    print(f"{'='*70}\n")

    # Check if checkpoint exists
    checkpoint_path = Path(MODEL_CHECKPOINT)
    if not checkpoint_path.exists():
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        print(f"Please train the model first using: python train.py")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Load test dataset (using validation set since no separate test set exists)
    print("Loading test dataset...")
    test_dataset = ForgeryDataset(
        f"{DATA_DIR}/test/images",
        f"{DATA_DIR}/test/masks",
        augmentation=get_augmentation('val')  # No augmentation for test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Test samples: {len(test_dataset)}\n")

    # Create model architecture (must match training)
    print("Loading model...")
    model = smp.FPN(
        encoder_name='resnext50_32x4d',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    )

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"✓ Loaded model from: {checkpoint_path}")
    print(f"✓ Model was trained for {checkpoint['epoch']} epochs")
    print(f"✓ Best validation IoU during training: {checkpoint['best_iou']:.4f}\n")

    # Evaluate on test set
    results, images, predictions, targets, sample_results = test_model(model, test_loader, device)

    # Print results
    print(f"\n{'='*70}")
    print(f"TEST SET RESULTS")
    print(f"{'='*70}")
    for metric, value in results.items():
        print(f"{metric:<20}: {value:.4f}")
    print(f"{'='*70}\n")

    # Save metrics report
    save_metrics_report(results, OUTPUT_DIR, checkpoint_path)

    # Visualize predictions
    print(f"\nGenerating visualizations...")
    visualize_predictions(images, predictions, targets, NUM_VISUALIZATIONS, OUTPUT_DIR)
    visualize_comparison_grid(images, predictions, targets, min(5, NUM_VISUALIZATIONS), OUTPUT_DIR)

    # Run best/worst analysis
    analyze_best_worst(sample_results, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print(f"TESTING COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - test_metrics.txt (detailed report)")
    print(f"  - test_metrics.csv (metrics in CSV format)")
    print(f"  - predictions_visualization.png (predictions vs ground truth)")
    print(f"  - detailed_comparison.png (TP/FP/FN overlay)")
    print(f"  - best_predictions.png (model's successes)")
    print(f"  - worst_predictions.png (detected but poor IoU)")
    print(f"  - missed_forgeries.png (false negatives)")
    print(f"  - true_negatives.png (correct authentic)")
    print(f"  - analysis_summary.txt (performance analysis)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
