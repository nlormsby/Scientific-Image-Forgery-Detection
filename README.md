# Scientific Image Forgery Detection

Deep learning-based detection of copy-move forgeries in scientific images using semantic segmentation.

## Overview

This project implements a systematic approach to detect image manipulation in scientific figures, particularly copy-move forgeries where regions are duplicated within an image. The system uses a **4-stage methodology** to identify the optimal architecture:

- **Stage 1**: Architecture comparison (U-Net, FPN, DeepLabV3+, etc.)
- **Stage 2**: Backbone comparison for winning architecture
- **Stage 3**: Ablation studies on architectural components
- **Stage 4**: Hyperparameter optimization

**Final Model**: FPN + ResNeXt50 encoder achieving **30% IoU** and **78% image-level accuracy**.

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch >= 2.0.0
- segmentation-models-pytorch
- albumentations (data augmentation)
- OpenCV, NumPy, Pandas, Matplotlib

## Dataset

Combined dataset from two sources:
- **Kaggle**: Scientific figure forgery dataset (forged + authentic images)
- **BioFors**: Biological/medical image forgery dataset (forged only)
- **Total**: ~5,000 images (70% train, 15% val, 15% test)

### Dataset Setup Instructions

**Note**: The `Datasets/` folder is in `.gitignore` and not included in this repository. You must download and prepare the datasets yourself.

#### Step 1: Download Datasets

1. **Kaggle Dataset**: Download the scientific figure forgery dataset from Kaggle
   - [Link to Kaggle dataset]

2. **BioFors Dataset**: Download the biological image forgery dataset
   - [Link to BioFors dataset]

#### Step 2: Organize Files

Create the following directory structure in your project root:

```
Datasets/
├── train_images/
│   ├── forged/          # Kaggle forged images
│   └── authentic/       # Kaggle authentic images
├── train_masks/         # Kaggle masks (.npy files)
├── biofors_images/      # BioFors images (numbered folders)
└── annotation_files/    # BioFors annotations (JSON)
    ├── idd_gt.json
    └── classification.json
```

**Specific instructions:**

1. **Kaggle dataset**:
   - Extract Kaggle forged images → `Datasets/train_images/forged/`
   - Extract Kaggle authentic images → `Datasets/train_images/authentic/`
   - Extract Kaggle mask files (.npy format) → `Datasets/train_masks/`
   - Masks are NumPy arrays stored as `.npy` files matching image names

2. **BioFors dataset**:
   - Extract BioFors images → `Datasets/biofors_images/`
   - Keep the numbered folder structure intact (e.g., `1/`, `738/`, `901/`, etc.)
   - Extract annotation JSON files → `Datasets/annotation_files/`
   - `idd_gt.json`: Contains bounding box coordinates `[x1, y1, x2, y2]` for forgeries
   - `classification.json`: Contains image type classifications (Blot/Gel, Microscopy, etc.)

#### Step 3: Verify Structure

Your `Datasets/` folder should look like:

```
Datasets/
├── train_images/
│   ├── forged/
│   │   ├── 12345.png
│   │   ├── 67890.png
│   │   └── ...
│   └── authentic/
│       ├── 11111.png
│       ├── 22222.png
│       └── ...
├── train_masks/
│   ├── 12345.npy      # NumPy array mask for 12345.png
│   ├── 67890.npy      # NumPy array mask for 67890.png
│   └── ...
├── biofors_images/
│   ├── 1/
│   │   ├── 010000.png
│   │   ├── 010001.png
│   │   └── ...
│   ├── 738/
│   │   ├── 010301.png
│   │   └── ...
│   ├── 901/
│   └── ...
└── annotation_files/
    ├── idd_gt.json           # BioFors bounding boxes
    └── classification.json   # BioFors image types
```

#### Step 4: Unify Datasets

First, combine Kaggle and BioFors into a unified format (converts BioFors JSON coordinates to PNG masks):

```bash
python unify_datasets.py
```

This creates `Datasets/unified/` with:
- Converted BioFors bounding boxes → binary PNG masks
- Converted Kaggle .npy masks → binary PNG masks
- Unified metadata.csv with all samples
- Initial train/val split

#### Step 5: Create Proper 3-Way Split

Then create the proper train/val/test split (70/15/15):

```bash
python create_proper_split.py
```

This reorganizes `Datasets/unified/` into proper train/val/test folders with stratified splitting.

## Quick Start (Reproduce Final Results)

### 1. Setup Dataset

Follow the **Dataset Setup Instructions** above to download and organize datasets, then:

```bash
# Unify Kaggle and BioFors datasets
python unify_datasets.py

# Create proper train/val/test split (70/15/15)
python create_proper_split.py
```

### 2. Train Final Optimized Model

```bash
python train.py
```

Configuration in `train.py`:
- `MULTI_MODEL_MODE = False` (single model training)
- `EXPERIMENT_NAME = "final_model_optimized"`
- `MAX_SAMPLES = None` (use full dataset)
- Model: FPN + ResNeXt50, batch_size=2, Adam optimizer

Training outputs:
- `experiments/final_model_optimized/best_model.pth`
- `experiments/final_model_optimized/history.csv`
- `experiments/final_model_optimized/curves.png`

### 3. Test Model

```bash
python test_model.py
```

Generates comprehensive analysis:
- Test metrics (IoU, Dice, Precision, Recall)
- Random prediction visualizations
- Best/worst predictions analysis
- Missed forgeries (False Negatives)
- True negatives (Correct authentic)
- Performance summary statistics

All results saved to `experiments/final_model_optimized/test_results/`.

## Running Your Own Experiments

### Stage 1: Architecture Comparison

Edit `train.py`:

```python
MULTI_MODEL_MODE = True
MODELS_TO_COMPARE = [
    {'name': 'unet_resnet34', 'model': lambda: smp.Unet(encoder_name='resnet34', ...)},
    {'name': 'fpn_resnet34', 'model': lambda: smp.FPN(encoder_name='resnet34', ...)},
    # Add more architectures
]
```

Run: `python train.py`

Results: `experiments/leaderboard.csv` and `experiments/model_comparison.png`

### Stage 2: Backbone Comparison

```python
MODELS_TO_COMPARE = [
    {'name': 'fpn_resnet50', 'model': lambda: smp.FPN(encoder_name='resnet50', ...)},
    {'name': 'fpn_resnext50', 'model': lambda: smp.FPN(encoder_name='resnext50_32x4d', ...)},
    {'name': 'fpn_efficientnet_b4', 'model': lambda: smp.FPN(encoder_name='efficientnet-b4', ...)},
]
```

### Stage 3: Ablation Studies

```python
from Models.fpn import create_fpn_full, create_fpn_no_lateral, create_fpn_no_topdown

MODELS_TO_COMPARE = [
    {'name': 'fpn_full', 'model': lambda: create_fpn_full(backbone_name='resnext50_32x4d')},
    {'name': 'fpn_no_lateral', 'model': lambda: create_fpn_no_lateral(backbone_name='resnext50_32x4d')},
    {'name': 'fpn_no_topdown', 'model': lambda: create_fpn_no_topdown(backbone_name='resnext50_32x4d')},
]
```

### Stage 4: Hyperparameter Tuning

Use `hyperparam_tuning.py` for systematic hyperparameter search:

```bash
python hyperparam_tuning.py
```

Tests combinations of:
- Loss functions (BCE, Dice, BCE+Dice, Focal, Tversky)
- Learning rates (1e-3, 5e-4, 1e-4, 5e-5)
- Batch sizes (2, 4, 8)
- Optimizers (Adam, AdamW, SGD)
- Schedulers (Cosine, Step, Plateau)
- Augmentation settings

Results: `experiments/stage4_results.csv`

## Project Structure

```
├── train.py                    # Main training script
├── test_model.py              # Model evaluation with analysis
├── hyperparam_tuning.py       # Phase 4 hyperparameter search
├── create_proper_split.py     # Dataset 3-way splitting
├── dataset.py                 # Dataset and augmentation
├── utils.py                   # Metrics and utilities
├── Models/
│   ├── fpn.py                # Custom FPN for ablation studies
│   ├── unet.py               # Custom U-Net variants
│   └── deeplabv3plus.py      # Custom DeepLabV3+
├── Datasets/
│   └── unified/              # Processed dataset (created by script)
│       ├── train/
│       ├── val/
│       └── test/
└── experiments/              # Training outputs
    └── [experiment_name]/
        ├── best_model.pth
        ├── history.csv
        └── curves.png
```

## Key Configuration Options

In `train.py`:

```python
# Training mode
MULTI_MODEL_MODE = False        # True: compare models, False: train single model
MAX_SAMPLES = None              # Limit dataset size (None = full dataset)

# Model settings
MODEL = smp.FPN(...)           # Choose architecture
BATCH_SIZE = 2                 # Batch size
OPTIMIZER = "adam"             # Optimizer: adam, adamw, sgd
LEARNING_RATE = 1e-4           # Learning rate
SCHEDULER = "cosine"           # LR scheduler: cosine, step, plateau
LOSS = "bce_dice"              # Loss: bce, dice, bce_dice, focal, tversky
AUGMENTATION = True            # Use data augmentation
EPOCHS = 30                    # Training epochs
```

## Results Summary

### Stage Results
- **Stage 1** (Architecture): FPN wins with 0.2141 IoU
- **Stage 2** (Backbone): ResNeXt50 wins with 0.2659 IoU
- **Stage 3** (Ablation): Top-down pathway critical (-53.9% when removed)
- **Stage 4** (Hyperparameters): batch_size=2 (+19.7%), Adam (+2.9%)

### Final Model Performance
- **IoU**: 30.15%
- **Dice**: 41.16%
- **Image Accuracy**: 78.16%
- **Precision**: 68.27%
- **Recall**: 39.29%

### Key Insights
- Top-down pathway in FPN is most critical component for multi-scale fusion
- Small batch sizes (2) significantly improve performance on this task
- Transfer learning from ImageNet provides strong feature extraction baseline
- Detection rate: Successfully detects majority of forgeries
- True negative rate: Reliably identifies authentic images

## Dataset Quality Check

Check masking consistency between Kaggle and BioFors:

```bash
python check_dataset_consistency.py
```

Verifies whether both datasets use the same masking convention (source+copied vs copied only).

## Citation

If you use this code, please cite the datasets:
- Kaggle Scientific Figure Forgery Dataset
- BioFors: Biological Image Forgery Dataset

## License

[Add your license here]