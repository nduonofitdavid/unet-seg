import os
from pathlib import Path
import random
import torch

D_PATH = os.getenv("DATASET_DIR", None)
if not D_PATH:
  raise Exception("Incomplete environment variable, DATASET_DIR not set")

DATASET_DIR = Path(D_PATH)
IMAGE_DIR = DATASET_DIR / 'images'
LABEL_DIR = DATASET_DIR / 'gtFine'
SAVE_DIR = Path('outputs')

SEED = 42
IMG_SIZE = (512, 1024)  
CROP_SIZE = (512, 512)  
BATCH_SIZE = 2
NUM_WORKERS = 2 
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = DEVICE.type == 'cuda'
USE_AMP = DEVICE.type == 'cuda'
IGNORE_INDEX = 255
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

random.seed(SEED)
torch.manual_seed(SEED)
print(f'Device: {DEVICE}, mixed precision: {USE_AMP}')

CLASS_NAMES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
CATEGORY_NAMES = ('flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle')
NUM_CLASSES = len(CLASS_NAMES)
VALID_LABEL_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
LABEL_TO_TRAINID = torch.full((256,), IGNORE_INDEX, dtype=torch.long)
LABEL_TO_TRAINID[VALID_LABEL_IDS] = torch.arange(NUM_CLASSES)
TRAINID_TO_CATID = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6])
