import random
from pathlib import Path
from constants import IGNORE_INDEX, MEAN, CROP_SIZE, IMG_SIZE, STD, LABEL_TO_TRAINID, SEED, IMAGE_DIR, BATCH_SIZE, PIN_MEMORY, NUM_WORKERS, LABEL_DIR

import torch  
from torchvision.transforms import InterpolationMode, v2
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torch.utils.data import DataLoader, Dataset

def get_pairs(image_dir, mask_dir):
  image_dir, mask_dir = Path(image_dir), Path(mask_dir)
  for directory in (image_dir, mask_dir):
    if not directory.is_dir():
      raise FileNotFoundError(f'Dataset directory does not exist: {directory}')

  images = sorted(path for path in image_dir.glob('*/*_leftImg8bit.png') if path.is_file())
  if not images:
    raise ValueError(f'No *_leftImg8bit.png images found under {image_dir}')

  pairs = []
  for image in images:
    stem = image.name.removesuffix('_leftImg8bit.png')
    mask = mask_dir / image.parent.name / f'{stem}_gtFine_labelIds.png'
    if not mask.is_file():
      raise FileNotFoundError(f'Missing mask for {image}: {mask}')
    pairs.append((image, mask))
  return pairs

def get_transform(train=False):
  transforms = [v2.Resize(IMG_SIZE, antialias=True)]
  if train:
    transforms += [
      v2.RandomCrop(CROP_SIZE),
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomRotation(
        degrees=15,
        interpolation=InterpolationMode.BILINEAR,
        fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE_INDEX},
      ),
      v2.ColorJitter(brightness=0.5, saturation=0.5, hue=0.3),
    ]
  transforms += [
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
    v2.ToPureTensor(),
  ]
  return v2.Compose(transforms)

class CityScapesDataset(Dataset):
  def __init__(self, pairs, transform=None):
    self.pairs = pairs
    self.transform = transform

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    image_path, mask_path = self.pairs[idx]
    image = decode_image(str(image_path), mode=ImageReadMode.RGB)
    mask = decode_image(str(mask_path))
    if mask.shape[0] != 1:
      raise ValueError(f'Expected a single-channel labelIds mask: {mask_path}')
    if image.shape[-2:] != mask.shape[-2:]:
      raise ValueError(f'Image and mask dimensions differ: {image_path}, {mask_path}')

    image = tv_tensors.Image(image)
    mask = tv_tensors.Mask(LABEL_TO_TRAINID[mask.squeeze(0).long()])
    if self.transform is not None:
      image, mask = self.transform(image, mask)
    return image, mask


train_pairs = get_pairs(IMAGE_DIR / 'train', LABEL_DIR / 'train')
if (IMAGE_DIR / 'val').exists() or (LABEL_DIR / 'val').exists():
  val_pairs = get_pairs(IMAGE_DIR / 'val', LABEL_DIR / 'val')
  print('Using the official train/val split.')
else:
  if len(train_pairs) < 2:
    raise ValueError('At least two image/mask pairs are needed for an 80/20 split.')
  random.Random(SEED).shuffle(train_pairs)
  val_count = max(1, int(len(train_pairs) * 0.2))
  val_pairs, train_pairs = train_pairs[:val_count], train_pairs[val_count:]
  print('Official val split absent; using a seeded 80/20 split of training pairs.')

train_ds = CityScapesDataset(train_pairs, transform=get_transform(train=True))
val_ds = CityScapesDataset(val_pairs, transform=get_transform())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
print(f'Training pairs: {len(train_ds)}, validation pairs: {len(val_ds)}')
