from constants import NUM_CLASSES, TRAINID_TO_CATID, CATEGORY_NAMES

import torch

def compute_confusion_matrix(preds, masks, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
  valid = masks != ignore_index
  indices = masks[valid].long() * num_classes + preds[valid].long()
  return torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)

def iou_from_confusion(confusion):
  true_positive = confusion.diag().double()
  union = confusion.sum(dim=0) + confusion.sum(dim=1) - true_positive
  return true_positive / union  

def compute_metrics(confusion):
  confusion = confusion.detach().cpu()
  class_iou = iou_from_confusion(confusion)
  num_categories = len(CATEGORY_NAMES)
  category_indices = TRAINID_TO_CATID[:, None] * num_categories + TRAINID_TO_CATID[None, :]
  category_confusion = torch.zeros(num_categories ** 2, dtype=torch.int64)
  category_confusion.scatter_add_(0, category_indices.flatten(), confusion.flatten())
  category_iou = iou_from_confusion(category_confusion.reshape(num_categories, num_categories))
  
  return {
    'accuracy': (confusion.diag().sum().double() / confusion.sum()).item(),
    'class_iou': class_iou,
    'mean_class_iou': class_iou.nanmean().item(),
    'category_iou': category_iou,
    'mean_category_iou': category_iou.nanmean().item(),
  }
