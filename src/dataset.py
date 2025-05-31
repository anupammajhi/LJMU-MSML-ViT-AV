from torch.utils.data import Dataset, DataLoader
from utils import make_transforms, stratification
from glob import glob
import os
import cv2
import torch
from sklearn.model_selection import train_test_split
from vars import CLASS2IDX

class KITTIDet(Dataset):
    # Class variables to store the split
    _train_files = None
    _val_files = None
    _split_initialized = False
    _class_weights = None

    def __init__(self, root, img_size, split="train", test_size=0.1, random_state=42):
        # Get all image files from training folder
        all_img_files = sorted(glob(os.path.join(root, "images",  "*.png")))
        self.label_dir = os.path.join(root,"labels")

        if not KITTIDet._split_initialized:
            KITTIDet._train_files, KITTIDet._val_files = train_test_split(
                all_img_files,
                test_size=test_size,
                random_state=random_state,
                stratify=stratification(all_img_files, self.label_dir)
            )
            KITTIDet._split_initialized = True
            print(f"[DATASET] Created train-val split: {len(KITTIDet._train_files)} train, {len(KITTIDet._val_files)} val")

        self.img_files = KITTIDet._train_files if split == "train" else KITTIDet._val_files
        self.transforms = make_transforms(train=(split == "train"), img_size=img_size)
        print(f"[DATASET] {split} set size: {len(self.img_files)}")

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

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
        h, w = img.shape[:2]

        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".png", ".txt"))
        boxes, labels = self._load_boxes(label_path)

        if len(boxes) == 0:
            center_x, center_y = w / 2, h / 2
            size = min(w, h) / 10 
            dummy_box = [center_x - size/2, center_y - size/2, center_x + size/2, center_y + size/2]
            boxes = [dummy_box]
            labels = [0]  

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
            "orig_size": torch.tensor([h, w]),
        }
        return img, target

    def __len__(self):
        return len(self.img_files)

def collate(batch):
    imgs, targets = list(zip(*batch))
    valid_entries = [(img, target) for img, target in zip(imgs, targets) if target["boxes"].shape[0] > 0]

    if len(valid_entries) == 0:
        return None

    valid_imgs, valid_targets = list(zip(*valid_entries))
    return torch.stack(valid_imgs), list(valid_targets)