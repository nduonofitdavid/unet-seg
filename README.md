# U-Net for Cityscapes Segmentation

A PyTorch implementation of U-Net for semantic segmentation of Cityscapes street scenes. The model assigns each pixel to one of 19 classes, including roads, buildings, pedestrians, and vehicles.

The repository provides image and mask loading, data augmentation, training, validation, and model checkpointing. The network uses convolutional encoder and decoder blocks with skip connections and learns from scratch.

## Setup

Create a Python environment and install compatible versions of PyTorch and torchvision:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision
```

The code uses torchvision transforms v2 and `decode_image`, along with `torch.amp`; use recent, compatible releases. For GPU training, install a PyTorch build compatible with your CUDA environment. CUDA is selected automatically when available; otherwise, training uses the CPU.

## Dataset

Provide Cityscapes RGB images and fine annotation masks in this layout:

```text
dataset/
├── images/
│   ├── train/<city>/*_leftImg8bit.png
│   └── val/<city>/*_leftImg8bit.png
└── gtFine/
    ├── train/<city>/*_gtFine_labelIds.png
    └── val/<city>/*_gtFine_labelIds.png
```

Image and mask filenames must share the same prefix and city folder. Use the original `labelIds` masks: the loader maps the 19 supported label IDs to training IDs `0–18` and ignores other labels with value `255`. If your images are extracted into a `leftImg8bit` directory, rename it to `images` or adjust `IMAGE_DIR` in `constants.py`.

Set the dataset root before running the code:

```bash
export DATASET_DIR=/absolute/path/to/dataset
```

The loader uses the official validation split when present. If both validation directories are absent, it creates a seeded 80/20 split from the training pairs. Missing masks or incomplete validation directories raise an error.

## Training

**Current prerequisite:** `metrics.py` references `IGNORE_INDEX` without importing it. Add it to that file's import from `constants` before starting training:

```python
from constants import NUM_CLASSES, TRAINID_TO_CATID, CATEGORY_NAMES, IGNORE_INDEX
```

Then run:

```bash
python train.py
```

Training settings are defined in `constants.py`:

| Setting | Default |
| --- | --- |
| Epochs | 50 |
| Batch size | 2 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Image resize (height × width) | 512 × 1024 |
| Training crop | 512 × 512 |
| Data loader workers | 2 |
| Random seed | 42 |

Training applies random crops, horizontal flips, rotations, and color jitter. Validation uses resized images without random augmentation. Both use ImageNet mean and standard deviation for normalization.

The model is optimized with AdamW and cross-entropy loss, excluding ignored pixels. CUDA training enables automatic mixed precision and gradient scaling.

## Metrics and output

Each epoch reports training and validation loss, pixel accuracy, mean class intersection over union (IoU), and mean category IoU. The seven categories are flat, construction, object, nature, sky, human, and vehicle. Classes or categories with zero union are excluded from their respective mean IoU.

The best weights, selected by validation mean class IoU, are saved to:

```text
outputs/unet_city.pth
```

This checkpoint contains the model's `state_dict` only. Each run starts fresh; optimizer state, automatic resumption, and a standalone inference command are not provided. Per-class and per-category IoU values printed after training describe the final validation epoch.

## Repository files

| File | Purpose |
| --- | --- |
| `constants.py` | Dataset paths, hyperparameters, device selection, and label mappings |
| `dataloader.py` | Image/mask pairing, preprocessing, augmentation, and data loaders |
| `models.py` | U-Net architecture |
| `metrics.py` | Confusion matrix, pixel accuracy, and class/category IoU |
| `train.py` | Model shape check, training, validation, and checkpoint saving |
