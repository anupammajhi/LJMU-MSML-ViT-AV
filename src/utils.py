import torch
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import make_baseline_vit, make_mobilevit, make_efficientvit, HybridCNNViT
from vars import CLASS2IDX,IMG_SIZE,NUM_CLASSES

class_names = {v: k for k, v in CLASS2IDX.items()}

def make_transforms(img_size, train: bool = True, ):
    if train:
        return A.Compose([
            A.RandomScale(scale_limit=0.3, p=0.2),
            A.Resize(height=img_size//1.5, width=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.ISONoise(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
            ], p=0.6),
            A.HorizontalFlip(p=0.2),
            A.Rotate(limit=10, p=0.2),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(height=img_size//1.5, width=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def stratification(all_img_files, label_dir):
  strata = []
  for img_path in all_img_files:
      label_path = os.path.join(label_dir, os.path.basename(img_path).replace(".png", ".txt"))
      has_objects = False
      try:
          with open(label_path) as f:
              for line in f:
                  cls, *_ = line.split()
                  if cls in CLASS2IDX:
                      has_objects = True
                      break
      except FileNotFoundError:
            has_objects = False
      strata.append(1 if has_objects else 0)

  return np.array(strata)

def get_model(name):
    print(f"[INFO] Initializing {name} model...")
    models = {
        "vit": make_baseline_vit,
        "mobilevit": make_mobilevit,
        "efficientvit": make_efficientvit,
        "hybridvit": HybridCNNViT
    }
    return models[name]()

def calculate_class_weights(dataset, num_classes):
    class_counts = torch.zeros(num_classes + 1)

    for i in range(len(dataset)):
        target = dataset[i][1]
        labels = target['labels']

        for label in labels:
            class_counts[label.item()] += 1

    for class_id, count in enumerate(class_counts):
        if class_id in class_names:
            print(f"Class {class_names[class_id]}: {count} occurrences")

    object_class_counts = class_counts[1:] # To ignore the background
    object_class_counts = torch.clamp(object_class_counts, min=1.0)

    inv_frequencies = 1.0 / object_class_counts

    normalized_class_weights = inv_frequencies * (num_classes / inv_frequencies.sum())

    final_class_weights = torch.zeros(num_classes + 1) # Add Background index back to set it to 0
    final_class_weights[1:] = normalized_class_weights 
    final_class_weights[0] = 0.0 

    return final_class_weights

def apply_class_weighted_loss(loss_dict, targets, class_weights, device):
    weighted_loss_dict = loss_dict.copy()

    if 'loss_classifier' in weighted_loss_dict:
        all_labels = torch.cat([t['labels'] for t in targets])

        class_weight_sum = 0
        class_counts = torch.zeros(len(class_weights), device=device)

        for label in all_labels:
            class_counts[label] += 1

        for class_id, count in enumerate(class_counts):
            if count > 0:
                class_weight_sum += class_weights[class_id] * count

        batch_weight = class_weight_sum / len(all_labels) if len(all_labels) > 0 else 1.0

        weighted_loss_dict['loss_classifier'] = weighted_loss_dict['loss_classifier'] * batch_weight

    return sum(weighted_loss_dict.values())

def visualize_detections(image, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
                         img_id, epoch, save_dir="visualizations"):

    Path(save_dir).mkdir(exist_ok=True)

    img_np = image.cpu().permute(1, 2, 0).numpy()

    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    ax.imshow(img_np)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(CLASS2IDX)+1))

    score_threshold = 0.3
    high_conf_indices = (pred_scores > score_threshold).nonzero().squeeze(1)

    for idx in high_conf_indices[:20]: 
        box = pred_boxes[idx].cpu().numpy()
        label = pred_labels[idx].item()
        score = pred_scores[idx].item()

        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=to_rgba(colors[label], 0.9),
            facecolor='none',
        )
        ax.add_patch(rect)

        class_name = [k for k, v in CLASS2IDX.items() if v == label][0]
        plt.text(x1, y1-5, f"{class_name}: {score:.2f}",
                 color=colors[label], fontsize=9)

    plt.title(f"Detection Results - Epoch {epoch}")
    plt.axis('off')

    save_path = os.path.join(save_dir, f"detection_epoch{epoch}_img{img_id}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"[VISUAL] Saved detection visualization to {save_path}")

class MetricsTracker:
  def __init__(self, model_name, output_dir="metrics"):
      self.model_name = model_name
      self.output_dir = Path(output_dir)
      self.output_dir.mkdir(exist_ok=True)

      self.epochs = []
      self.train_losses = []
      self.map_values = []
      self.mar_values = []
      self.per_class_maps = {}
      self.per_class_mars = {}
      self.learning_rates = []

  def update(self, epoch, train_loss, metrics_dict, lr):
      self.epochs.append(epoch)
      self.train_losses.append(train_loss)
      self.map_values.append(float(metrics_dict['map']))
      self.mar_values.append(float(metrics_dict['mar_100']))
      self.learning_rates.append(lr)

      if 'map_per_class' in metrics_dict and 'mar_100_per_class' in metrics_dict:
          for class_id, class_name in class_names.items():
              if class_name not in self.per_class_maps:
                  self.per_class_maps[class_name] = []
                  self.per_class_mars[class_name] = []

              if class_id < len(metrics_dict['map_per_class']):
                  map_val = float(metrics_dict['map_per_class'][class_id])
                  mar_val = float(metrics_dict['mar_100_per_class'][class_id])

                  if map_val >= 0 and mar_val >= 0:
                      self.per_class_maps[class_name].append(map_val)
                      self.per_class_mars[class_name].append(mar_val)
                  else:
                      prev_map = self.per_class_maps[class_name][-1] if self.per_class_maps[class_name] else 0
                      prev_mar = self.per_class_mars[class_name][-1] if self.per_class_mars[class_name] else 0
                      self.per_class_maps[class_name].append(prev_map)
                      self.per_class_mars[class_name].append(prev_mar)

  def plot_loss_curve(self):
      plt.figure(figsize=(10, 6))
      plt.plot(self.epochs, self.train_losses, 'b-', marker='o', label='Training Loss')
      plt.title(f'Training Loss - {self.model_name}')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.grid(True, linestyle='--', alpha=0.7)
      plt.legend()
      save_path = self.output_dir / f"{self.model_name}_loss_curve.png"
      plt.savefig(save_path, bbox_inches='tight', dpi=150)
      plt.close()
      print(f"[VISUAL] Saved loss curve to {save_path}")

  def plot_map_progress(self):
      plt.figure(figsize=(10, 6))
      plt.plot(self.epochs, self.map_values, 'r-', marker='o', label='mAP')
      plt.plot(self.epochs, self.mar_values, 'b-', marker='^', label='mAR@100')

      f1_values = [2 * (ap * ar) / (ap + ar + 1e-16) for ap, ar in zip(self.map_values, self.mar_values)]
      plt.plot(self.epochs, f1_values, 'm-', marker='*', label='F1 Score')

      plt.title(f'Detection Performance Metrics - {self.model_name}')
      plt.xlabel('Epoch')
      plt.ylabel('Metric Value')
      plt.grid(True, linestyle='--', alpha=0.7)
      plt.legend()
      save_path = self.output_dir / f"{self.model_name}_map_progress.png"
      plt.savefig(save_path, bbox_inches='tight', dpi=150)
      plt.close()
      print(f"[VISUAL] Saved mAP progress plot to {save_path}")

  def plot_per_class_performance(self, final_epoch=True):
      if not self.per_class_maps:
        print("[WARNING] No per-class metrics available for visualization")
        return

      if not final_epoch and self.epochs[-1] != max(self.epochs):
        return

      classes = []
      map_values = []
      mar_values = []
      f1_values = []

      for class_name in self.per_class_maps:
        if self.per_class_maps[class_name] and self.per_class_mars[class_name]:
          classes.append(class_name)
          map_val = self.per_class_maps[class_name][-1]
          mar_val = self.per_class_mars[class_name][-1]
          f1_val = 2 * (map_val * mar_val) / (map_val + mar_val + 1e-16)

          map_values.append(map_val)
          mar_values.append(mar_val)
          f1_values.append(f1_val)

      sorted_indices = np.argsort(map_values)[::-1]  # descending
      classes = [classes[i] for i in sorted_indices]
      map_values = [map_values[i] for i in sorted_indices]
      mar_values = [mar_values[i] for i in sorted_indices]
      f1_values = [f1_values[i] for i in sorted_indices]

      plt.figure(figsize=(12, 8))
      x = np.arange(len(classes))
      width = 0.25

      plt.bar(x - width, map_values, width, label='mAP', color='#4285F4')
      plt.bar(x, mar_values, width, label='mAR', color='#EA4335')
      plt.bar(x + width, f1_values, width, label='F1', color='#34A853')

      plt.xlabel('Class')
      plt.ylabel('Score')
      plt.title(f'Per-Class Performance - {self.model_name}')
      plt.xticks(x, classes, rotation=45, ha='right')
      plt.legend()
      plt.grid(True, linestyle='--', alpha=0.3, axis='y')
      plt.tight_layout()

      save_path = self.output_dir / f"{self.model_name}_per_class_performance.png"
      plt.savefig(save_path, bbox_inches='tight', dpi=150)
      plt.close()
      print(f"[VISUAL] Saved per-class performance plot to {save_path}")