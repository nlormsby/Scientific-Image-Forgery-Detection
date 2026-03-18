"""
Training Configuration

Edit this file to change training parameters.
No need for command line arguments - just modify the values below and run train.py
"""

# ==============================================================================
# EXPERIMENT SETTINGS
# ==============================================================================

# Experiment name (automatically timestamped)
EXPERIMENT_NAME = "deeplabv3plus_experiment"

# Random seed for reproducibility
SEED = 42

# ==============================================================================
# DATA SETTINGS
# ==============================================================================

# Path to unified dataset
DATA_DIR = "Datasets/unified"

# Maximum number of training samples (None = use all data)
# Set to a small number like 500 for quick testing
MAX_SAMPLES = None  # Options: None, 100, 500, 1000, 2000, etc.

# ==============================================================================
# MODEL SETTINGS
# ==============================================================================

# Model architecture options:
# - 'unet' - Standard U-Net
# - 'unetplusplus' or 'unet++' - U-Net++
# - 'deeplabv3+' - DeepLabV3+
# - 'fpn' - Feature Pyramid Network
# - 'pspnet' - PSPNet
# - 'manet' - Multi-scale Attention Net
# - 'custom_deeplabv3plus' - Custom DeepLabV3+ from Models/
ARCHITECTURE = "custom_deeplabv3plus"

# DeepLabV3+ variant (only for custom_deeplabv3plus)
# Options: 'default', 'no_aspp', 'no_skip', 'minimal'
DEEPLABV3PLUS_VARIANT = "default"

# Encoder backbone options:
# - 'resnet34', 'resnet50', 'resnet101'
# - 'efficientnet-b0', 'efficientnet-b3', 'efficientnet-b7'
# - 'mobilenet_v2'
# For custom DeepLabV3+: 'resnet34' or 'resnet50' only
ENCODER = "resnet34"

# Use ImageNet pretrained weights
PRETRAINED = True

# ==============================================================================
# TRAINING SETTINGS
# ==============================================================================

# Number of epochs
EPOCHS = 50

# Batch size (reduce if out of memory)
BATCH_SIZE = 8

# Learning rate
LEARNING_RATE = 0.001

# Optimizer: 'adam', 'adamw', 'sgd'
OPTIMIZER = "adamw"

# Weight decay (L2 regularization)
WEIGHT_DECAY = 1e-4

# Learning rate scheduler: 'cosine', 'step', 'plateau', 'none'
SCHEDULER = "cosine"

# Loss function: 'bce', 'dice', 'bce_dice', 'focal', 'tversky'
LOSS = "bce_dice"

# ==============================================================================
# AUGMENTATION SETTINGS
# ==============================================================================

# Enable data augmentation
AUGMENTATION = True

# ==============================================================================
# SYSTEM SETTINGS
# ==============================================================================

# Number of data loading workers
NUM_WORKERS = 4

# Output directory for checkpoints and logs
OUTPUT_DIR = "experiments"

# ==============================================================================
# QUICK PRESETS
# ==============================================================================

# Uncomment one of these presets for common configurations:

# # QUICK TEST (5 minutes)
# MAX_SAMPLES = 100
# EPOCHS = 3
# BATCH_SIZE = 4

# # MEDIUM TEST (30-60 minutes)
# MAX_SAMPLES = 1000
# EPOCHS = 10
# BATCH_SIZE = 8

# # FULL TRAINING
# MAX_SAMPLES = None
# EPOCHS = 50
# BATCH_SIZE = 8

# ==============================================================================
# ABLATION STUDY PRESETS
# ==============================================================================

# DeepLabV3+ Ablation Studies:
# Uncomment one at a time to compare different variants

# # Experiment 1: Full DeepLabV3+ (baseline)
# ARCHITECTURE = "custom_deeplabv3plus"
# DEEPLABV3PLUS_VARIANT = "default"
# EXPERIMENT_NAME = "deeplabv3plus_full"

# # Experiment 2: No ASPP module
# ARCHITECTURE = "custom_deeplabv3plus"
# DEEPLABV3PLUS_VARIANT = "no_aspp"
# EXPERIMENT_NAME = "deeplabv3plus_no_aspp"

# # Experiment 3: No skip connections
# ARCHITECTURE = "custom_deeplabv3plus"
# DEEPLABV3PLUS_VARIANT = "no_skip"
# EXPERIMENT_NAME = "deeplabv3plus_no_skip"

# # Experiment 4: Minimal (no ASPP, no skip)
# ARCHITECTURE = "custom_deeplabv3plus"
# DEEPLABV3PLUS_VARIANT = "minimal"
# EXPERIMENT_NAME = "deeplabv3plus_minimal"

# ==============================================================================
# ENCODER BACKBONE COMPARISON
# ==============================================================================

# # ResNet34 (lightweight, fast)
# ENCODER = "resnet34"
# EXPERIMENT_NAME = "deeplabv3plus_resnet34"

# # ResNet50 (more capacity)
# ENCODER = "resnet50"
# EXPERIMENT_NAME = "deeplabv3plus_resnet50"
