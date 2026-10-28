from constants import DEVICE, PIN_MEMORY, NUM_CLASSES, IGNORE_INDEX, USE_AMP, SAVE_DIR, NUM_EPOCHS, CLASS_NAMES, CATEGORY_NAMES, LEARNING_RATE, WEIGHT_DECAY, SEED
from metrics import compute_confusion_matrix, compute_metrics
from models import UNET
from dataloader import train_loader, val_loader

import torch
from torch import nn
from torch.optim import AdamW

def run_epoch(model, loader, loss_fn, optimizer=None, scaler=None):
  training = optimizer is not None
  if training and USE_AMP and scaler is None:
    raise ValueError('CUDA mixed-precision training requires a GradScaler.')
  model.train(training)
  confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=DEVICE)
  total_loss = 0.0
  total_valid_pixels = 0

  for images, masks in loader:
    images = images.to(DEVICE, non_blocking=PIN_MEMORY)
    masks = masks.to(DEVICE, non_blocking=PIN_MEMORY)
    valid_pixels = (masks != IGNORE_INDEX).sum().item()
    if valid_pixels == 0:
      continue
    if training:
      optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(training):
      with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
        logits = model(images)
        loss = loss_fn(logits, masks)
      if training:
        if scaler is None:
          loss.backward()
          optimizer.step()
        else:
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

    total_loss += loss.item() * valid_pixels
    total_valid_pixels += valid_pixels
    confusion += compute_confusion_matrix(logits.detach().argmax(dim=1), masks)

  if total_valid_pixels == 0:
    raise ValueError('This epoch contains no valid labeled pixels.')
  metrics = compute_metrics(confusion)
  metrics['loss'] = total_loss / total_valid_pixels
  return metrics

# sanity checks
with torch.no_grad():
  test_model = UNET(features=(4, 8, 16)).eval()
  test_input = torch.randn(1, 3, 33, 65)
  test_output = test_model(test_input)
  assert test_output.shape == (1, NUM_CLASSES, 33, 65)
print('U-Net shape check passed.')
del test_model, test_input, test_output

# start official training
torch.manual_seed(SEED)
model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
best_mclass_iou = -float('inf')


val_metrics = {}
for epoch in range(1, NUM_EPOCHS + 1):
  train_metrics = run_epoch(model, train_loader, loss_fn, optimizer=optimizer, scaler=scaler)
  val_metrics = run_epoch(model, val_loader, loss_fn)
  print(f'Epoch {epoch}/{NUM_EPOCHS}')

  for split, metrics in [('Train', train_metrics), ('Val', val_metrics)]:
    print(
      f"{split}: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}, "
      f"mean class IoU={metrics['mean_class_iou']:.4f}, "
      f"mean category IoU={metrics['mean_category_iou']:.4f}"
    )

  if val_metrics['mean_class_iou'] > best_mclass_iou:
    best_mclass_iou = val_metrics['mean_class_iou']
    torch.save(model.state_dict(), SAVE_DIR / 'unet_city.pth')
    print(f'Saved best weights with validation mean class IoU {best_mclass_iou:.4f}')
  print()


# inspect the last epoch
for names, scores in [(CLASS_NAMES, val_metrics['class_iou']), (CATEGORY_NAMES, val_metrics['category_iou'])]:
  for name, score in zip(names, scores):
    print(f'{name:>14}: {score.item():.4f}')
  print()
  