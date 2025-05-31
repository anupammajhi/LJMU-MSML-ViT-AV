import xml.etree.ElementTree as ET
import os, time
import gc
import psutil
import random
import timm, cv2
import math
import numpy as np
import pandas as pd
import torch, torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from collections import OrderedDict
from torchvision.models import mobilenet_v2
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from vars import CLASS2IDX
from utils import get_model

NUM_IMAGES = 10  # Number of images to process

def make_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size//1.5, width=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def collate_fn(batch):
    return tuple(zip(*batch))

class KITTIDataset(Dataset):
    __images__ = None

    def __init__(self, root_dir, img_size, transforms=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transforms = transforms

        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir,"labels")
        if KITTIDataset.__images__ is not None:
            self.image_files = KITTIDataset.__images__
        else:
            random.seed(42)
            basenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
            self.image_files = [os.path.join(self.image_dir, bn) for bn in basenames]
            self.image_files = random.sample(self.image_files, NUM_IMAGES)
            KITTIDataset.__images__ = self.image_files

        self.transforms = make_transforms(img_size=img_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR→RGB
        h, w = img.shape[:2]

        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".png", ".txt"))
        boxes, labels = self._load_boxes(label_path)

        if len(boxes) == 0:
            # If No valid boxes in this image after filtering, create a dummy box for continuity
            center_x, center_y = w / 2, h / 2
            size = min(w, h) / 10  # small box
            dummy_box = [center_x - size/2, center_y - size/2, center_x + size/2, center_y + size/2]
            boxes = [dummy_box]
            labels = [0]  # Use any valid class

        aug = self.transforms(image=img, bboxes=boxes, class_labels=labels)
        img, boxes, labels = aug["image"], aug["bboxes"], aug["class_labels"]
        img = img.float() / 255.0

        if len(boxes) == 0:
            dummy_box = [0, 0, 10, 10]
            boxes = [dummy_box]
            labels = [0]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return img, target

    def _load_boxes(self, label_path):
        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                cls, _, _, _, x1, y1, x2, y2, *_ = line.split()
                if cls == "DontCare":
                    continue
                width = float(x2) - float(x1)
                if width <= 20:
                    continue
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(CLASS2IDX[cls])
        return boxes, labels
    

def calculate_iou(box1, box2):
    # Determine coordinates of intersection rectangle
    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])

    # Compute area of intersection rectangle
    intersection_area = max(0, x2_intersect - x1_intersect) * max(0, y2_intersect - y1_intersect)

    # Compute area of both prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute intersection over union by dividing intersection area by union area
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / (union_area + 1e-6) # Also avoid division by zero

    return iou

def evaluate(model, dataloader, device):
    model.eval()
    results = []
    inference_times = []

    total_ground_truths_for_accuracy = 0
    correct_detections_top1 = 0
    correct_detections_top5 = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]

            # Measure inference time and resource usage
            process = psutil.Process(os.getpid())
            start_cpu_percent = process.cpu_percent(interval=None)
            start_memory = process.memory_info().rss
            start_time = time.time()

            outputs = model(images)

            end_time = time.time()
            end_cpu_percent = process.cpu_percent(interval=None)
            end_memory = process.memory_info().rss

            inference_times.append(end_time - start_time)
            cpu_usage = (end_cpu_percent + start_cpu_percent) / 2
            memory_usage = end_memory - start_memory

            for i, output in enumerate(outputs):
                img_id = targets[i]["image_id"].item()
                true_boxes = targets[i]["boxes"]
                true_labels = targets[i]["labels"]
                pred_boxes = output["boxes"]
                pred_labels = output["labels"]
                pred_scores = output["scores"]

                total_ground_truths_for_accuracy += len(true_boxes)
                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes_sorted = pred_boxes[sorted_indices]
                pred_labels_sorted = pred_labels[sorted_indices]
                pred_scores_sorted = pred_scores[sorted_indices]

                for k in range(len(true_boxes)):
                    true_box = true_boxes[k]
                    true_label = true_labels[k].item()

                    best_iou = 0
                    best_pred_idx = -1

                    # Find best predicted box that overlaps significantly with current true box
                    for j in range(len(pred_boxes_sorted)):
                        pred_box = pred_boxes_sorted[j]

                        iou = calculate_iou(pred_box, true_box)
                        iou_threshold = 0.5

                        if iou > iou_threshold and iou > best_iou:
                             best_iou = iou
                             best_pred_idx = j

                    # If a matching prediction is found for this ground truth box
                    if best_pred_idx != -1:
                         matched_pred_label = pred_labels_sorted[best_pred_idx].item()

                         # Check Top-1 Accuracy for this matched pair
                         if matched_pred_label == true_label:
                             correct_detections_top1 += 1

                         top5_labels_indices = torch.topk(pred_scores, k=min(5, len(pred_scores))).indices
                         top5_predicted_labels = pred_labels[top5_labels_indices].tolist()

                         if true_label in top5_predicted_labels:
                             correct_detections_top5 += 1

                # Calculate TP, FP, FN for mAP/mAR calculation based on sorted predictions
                matched_true_boxes_in_image = set()
                img_tps_list = []
                img_fps_list = []
                img_fns_count = len(true_boxes)

                for j in range(len(pred_boxes_sorted)):
                  pred_box = pred_boxes_sorted[j]
                  pred_label = pred_labels_sorted[j].item()
                  pred_score = pred_scores_sorted[j].item()

                  best_iou_for_tp = 0
                  best_match_idx_for_tp = -1

                  for k in range(len(true_boxes)):
                      true_box = true_boxes[k]
                      true_label = true_labels[k].item()

                      if k in matched_true_boxes_in_image:
                          continue

                      if pred_label == true_label:
                          iou = calculate_iou(pred_box, true_box)

                          iou_threshold_tp = 0.5
                          if iou > iou_threshold_tp and iou > best_iou_for_tp:
                              best_iou_for_tp = iou
                              best_match_idx_for_tp = k

                  if best_match_idx_for_tp != -1:
                      img_tps_list.append({"score": pred_score, "label": pred_label})
                      matched_true_boxes_in_image.add(best_match_idx_for_tp)
                      img_fns_count -= 1
                  else:
                      img_fps_list.append({"score": pred_score, "label": pred_label})


                results.append({
                    "image_id": img_id,
                    "true_boxes": true_boxes.cpu().numpy(),
                    "true_labels": true_labels.cpu().numpy(),
                    "pred_boxes": pred_boxes.cpu().numpy(),
                    "pred_labels": pred_labels.cpu().numpy(),
                    "pred_scores": pred_scores.cpu().numpy(),
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "tps": img_tps_list,
                    "fps": img_fps_list,
                    "fns": img_fns_count
                })


    all_tps = []
    all_fps = []
    all_fns = 0
    total_ground_truths_map_mar = 0

    for result in results:
        all_tps.extend(result["tps"])
        all_fps.extend(result["fps"])
        all_fns += result["fns"]
        total_ground_truths_map_mar += len(result["true_boxes"])

    all_predictions_for_map = sorted(all_tps + all_fps, key=lambda x: x["score"], reverse=True)

    cumulative_tps_map = 0
    cumulative_fps_map = 0

    precisions = []
    recalls = []

    for i, prediction in enumerate(all_predictions_for_map):
        if prediction in all_tps:
            cumulative_tps_map += 1
        else:
            cumulative_fps_map += 1

        precision = cumulative_tps_map / (cumulative_tps_map + cumulative_fps_map) if (cumulative_tps_map + cumulative_fps_map) > 0 else 0
        recall = cumulative_tps_map / total_ground_truths_map_mar if total_ground_truths_map_mar > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    average_precision = 0
    unique_recalls, unique_indices = np.unique(recalls, return_index=True)
    unique_precisions = [precisions[i] for i in unique_indices]

    if len(unique_recalls) > 1:
        average_precision = np.trapz(unique_precisions, unique_recalls)
    elif len(unique_recalls) == 1 and total_ground_truths_map_mar > 0:
         average_precision = unique_precisions[0] * unique_recalls[0]

    mean_average_recall = np.mean(recalls) if recalls else 0

    # Calculate Overall and Top-N Accuracy
    average_inference_time = np.mean(inference_times) if inference_times else 0
    overall_accuracy = (correct_detections_top1 / total_ground_truths_for_accuracy) if total_ground_truths_for_accuracy > 0 else 0
    top1_accuracy = (correct_detections_top1 / total_ground_truths_for_accuracy) if total_ground_truths_for_accuracy > 0 else 0
    top5_accuracy = (correct_detections_top5 / total_ground_truths_for_accuracy) if total_ground_truths_for_accuracy > 0 else 0

    average_cpu_usage = np.mean([r["cpu_usage"] for r in results]) if results else 0
    average_memory_usage = np.mean([r["memory_usage"] for r in results]) if results else 0

    metrics = {
        "Average Inference Time (s)": average_inference_time,
        "Overall Accuracy": overall_accuracy,
        "Top-1 Accuracy": top1_accuracy,
        "Top-5 Accuracy": top5_accuracy,
        "mAP": average_precision,
        "mAR": mean_average_recall,
        "Average CPU Usage (%)": average_cpu_usage,
        "Average Memory Usage (bytes)": average_memory_usage,
    }

    return metrics, results

if __name__ == "__main__":

    # Assuming already downloaded and extracted the KITTI dataset
    KITTI_ROOT = '../data/'
    MODEL_DIR = '../saved_models/'
    OUTPUT_DIR = 'evaluation_results'

    CLASS2IDX = {"Car": 1, "Van": 2, "Truck": 3, "Pedestrian": 4, "Person_sitting": 5, "Cyclist": 6, "Tram": 7, "Misc": 8}
    NUM_CLASSES = len(CLASS2IDX)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # device is always cpu, as our devices are resource constrained
    device = "cpu"

    # List saved models
    saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    print(f"Found {len(saved_models)} models: {saved_models}")

    all_model_metrics = []

    for model_name in saved_models:
        print(f"\nEvaluating model: {model_name}")
        model_path = os.path.join(MODEL_DIR, model_name)

        model_short_name = model_name.split('_')[0]

        if model_short_name.startswith('vit'):
            IMG_SIZE = 384
        elif model_short_name.startswith('mobilevit'):
            IMG_SIZE = 256
        elif model_short_name.startswith('efficientvit'):
            IMG_SIZE = 224
        elif "384" in model_name:
            IMG_SIZE = 384
        elif "266" in model_name:
            IMG_SIZE = 256
        elif "512" in model_name:
            IMG_SIZE = 512
        else:
            IMG_SIZE = 384

        print(f"INFO: Image Size: {IMG_SIZE}px")

        model = get_model(model_short_name)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue 

        # Evaluate the model
        dataset = KITTIDataset(root_dir=KITTI_ROOT, img_size=IMG_SIZE)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
        metrics, results = evaluate(model, dataloader, device)

        # Store metrics
        metrics["Model Name"] = model_name
        all_model_metrics.append(metrics)

        # Clean up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Save summary metrics to CSV
        metrics_df = pd.DataFrame(all_model_metrics)
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_summary.csv'), index=False)

        print("\n--- Model Performance Summary ---")
        print(metrics_df)

        # Generate plots
        print("\nGenerating plots...")

        # Plot Inference Time
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_df["Model Name"], metrics_df["Average Inference Time (s)"])
        plt.ylabel("Average Inference Time (s)")
        plt.title("Average Inference Time per Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'inference_time_plot.png'))
        plt.close()

        # Plot Top-1 Accuracy
        if 'Top-1 Accuracy' in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_df["Model Name"], metrics_df["Top-1 Accuracy"])
            plt.ylabel("Top-1 Accuracy")
            plt.title("Top-1 Accuracy per Model")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'top1_accuracy_plot.png'))
            plt.close()

        # Plot Top-5 Accuracy
        if 'Top-5 Accuracy' in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_df["Model Name"], metrics_df["Top-5 Accuracy"])
            plt.ylabel("Top-5 Accuracy")
            plt.title("Top-5 Accuracy per Model")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'top5_accuracy_plot.png'))
            plt.close()

        # Plot Top-5 Accuracy
        if 'Overall Accuracy' in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_df["Model Name"], metrics_df["Overall Accuracy"])
            plt.ylabel("Overall Accuracy")
            plt.title("Overall Accuracy per Model")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'overall_accuracy_plot.png'))
            plt.close()

        # Plot mAP
        if 'mAP' in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_df["Model Name"], metrics_df["mAP"])
            plt.ylabel("mAP")
            plt.title("mAP per Model")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'map_plot.png'))
            plt.close()

        # Plot mAR
        if 'mAR' in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_df["Model Name"], metrics_df["mAR"])
            plt.ylabel("mAR")
            plt.title("mAR per Model")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'mar_plot.png'))
            plt.close()


        # Plot Average CPU Usage
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_df["Model Name"], metrics_df["Average CPU Usage (%)"])
        plt.ylabel("Average CPU Usage (%)")
        plt.title("Average CPU Usage per Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'cpu_usage_plot.png'))
        plt.close()

        # Plot Average Memory Usage
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_df["Model Name"], metrics_df["Average Memory Usage (bytes)"])
        plt.ylabel("Average Memory Usage (bytes)")
        plt.title("Average Memory Usage per Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'memory_usage_plot.png'))
        plt.close()


        print(f"\nEvaluation complete. Results saved to '{OUTPUT_DIR}'")