import argparse
import time
from pathlib import Path
import numpy as np
import torch
from dataset import KITTIDet, collate
from vars import CLASS2IDX
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils import calculate_class_weights, get_model, apply_class_weighted_loss, MetricsTracker, visualize_detections, class_names
from vars import NUM_CLASSES

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--epochs", type=int, default=50)
  p.add_argument("--batch", type=int, default=25)      # 15 is a good number for small systems, 50 for big
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--model", type=str, choices=['vit', 'mobilevit', 'efficientvit', 'hybridvit'], default='vit')
  p.add_argument("--prune", action='store_true')
  p.add_argument("--qat", action='store_true')
  p.add_argument("--distill", action='store_true')

  try:
    args = p.parse_args([]) 
    print(f"[INFO] Running with arguments: {vars(args)}")
  except SystemExit as e:
    print(f"[WARNING] argparse could not parse arguments. Using default values.")
    args = argparse.Namespace(
        epochs=50,
        batch=4,
        lr=2e-4,
        model='vit',
        prune=False,
        qat=False,
        distill=False
    )

    print(f"[INFO] Using default arguments: {vars(args)}")

  global IMG_SIZE
  if args.model in ['vit']:
    IMG_SIZE = 384
  elif args.model in ['mobilevit']:
    IMG_SIZE = 256
  elif args.model in ['efficientvit']:
    IMG_SIZE = 224
  else:
    IMG_SIZE = 256  # Default for hybridvit

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"[SETUP] Using device: {device}")

  train_ds = KITTIDet(root='../data/', split='train', img_size=IMG_SIZE)
  test_ds = KITTIDet(root='../data/', split='val', img_size=IMG_SIZE)
  print(f"[INFO] Dataset sizes - Train: {len(train_ds)}, Val: {len(test_ds)}")

  best_metric_value = 0.0 
  best_epoch = 0
  model_save_dir = Path("saved_models")
  model_save_dir.mkdir(exist_ok=True)
  best_model_path = model_save_dir / f"{args.model}_best.pth"

  print("[SETUP] Creating dataloaders...")
  train_dl = torch.utils.data.DataLoader(
      train_ds,
      batch_size=args.batch,
      shuffle=True,
      num_workers=4,
      collate_fn=collate
  )
  test_dl = torch.utils.data.DataLoader(
      test_ds,
      batch_size=1,
      shuffle=False,
      num_workers=4,
      collate_fn=collate
  )

  print("[SETUP] Calculating class weights based on training data...")
  if KITTIDet._class_weights is not None:
    class_weights = KITTIDet._class_weights
  else:
    class_weights = calculate_class_weights(train_ds, NUM_CLASSES)
    KITTIDet._class_weights = class_weights

  class_weights = class_weights.to(device)
  print(f"[INFO] Class weights: {class_weights}")

  model = get_model(args.model).to(device)
  metrics_tracker = MetricsTracker(args.model)

  best_lr = 1e-4
  print(f"[SETUP] Using fixed learning rate of {best_lr}...")

  optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)  

  ### TRAINING LOOP
  print("\n[TRAINING] Starting training loop...")
  global_step = 0
  for epoch in range(args.epochs):
    print(f"\n[EPOCH] Starting epoch {epoch+1}/{args.epochs}")
    model.train()

    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    start_time = time.time()

    epoch_loss = 0
    batch_count = 0

    for imgs, targets in train_dl:
      batch_count += 1
      if batch_count % 50 == 0:
        print(f"[PROGRESS] Processing batch {batch_count}/{len(train_dl)}")

      if imgs is None:
        print("[WARNING] Skipping empty batch")
        continue

      imgs = imgs.to(device)
      tgs = [{k: v.to(device) for k, v in t.items()} for t in targets]

      valid_batch = all(t["boxes"].shape[0] > 0 for t in tgs)
      if not valid_batch:
          print("[WARNING] Skipping batch with empty boxes")
          continue

      # Forward Pass
      loss_dict = model(imgs, tgs)
      loss = apply_class_weighted_loss(loss_dict, tgs, class_weights, device)
      epoch_loss += loss.item()

      # Backward Pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      global_step += 1

    avg_epoch_loss = epoch_loss / batch_count
    print(f"[EPOCH] Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

    model.eval()

    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True,
                             iou_thresholds = [ 0.5 ]
                             ).to(device)

    with torch.no_grad():
      formatted_preds = []
      formatted_targets = []

      for i, (imgs, targets) in enumerate(test_dl):

        if i % 250 == 0:
            print(f"[EVAL] Evaluating batch {i}/{len(test_dl)}")
        imgs = imgs.to(device)
        preds = model(imgs)

        for pred in preds:
            formatted_preds.append({
                'boxes': pred['boxes'],
                'scores': pred['scores'],
                'labels': pred['labels']
            })

        formatted_targets.extend([{k: v.to(device) for k, v in t.items()} for t in targets])

        if i == 0:
            print(f"Sample prediction structure: {preds[0].keys()}")
            print(f"Boxes shape: {preds[0]['boxes'].shape}")
            print(f"Number of detections: {len(preds[0]['boxes'])}")
            print(f"Sample scores: {preds[0]['scores'][:5]}") 
            print(f"  Labels: {preds[0]['labels'][:5]}") 

            score_threshold = 0.05 
            high_conf_preds = preds[0]['scores'] > score_threshold

            print(f"  Predictions with score > {score_threshold}: {high_conf_preds.sum().item()}")
            print(f"  Ground truth boxes shape: {targets[0]['boxes'].shape}")
            print(f"  Ground truth labels: {targets[0]['labels']}")

      m.update(formatted_preds, formatted_targets)

    metrics = m.compute()

    # Save Best Model
    current_metric_value = metrics['map'].item()
    print(f"[EVAL] Epoch {epoch+1} - mAP: {current_metric_value:.4f}")

    if current_metric_value > best_metric_value:
        best_metric_value = current_metric_value
        best_epoch = epoch + 1
        print(f"[SAVE] New best model found at Epoch {best_epoch} with mAP: {best_metric_value:.4f}. Saving model...")
        torch.save(model.state_dict(), best_model_path)

    f1_score = 2 * (metrics['map'] * metrics['mar_100']) / (metrics['map'] + metrics['mar_100'] + 1e-16)

    current_lr = optimizer.param_groups[0]['lr']

    metrics_tracker.update(epoch + 1, avg_epoch_loss, metrics, current_lr)

    if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
      metrics_tracker.plot_loss_curve()
      metrics_tracker.plot_map_progress()
      metrics_tracker.plot_per_class_performance(final_epoch=(epoch == args.epochs - 1))

    with torch.no_grad():
      vis_indices = np.random.choice(len(test_ds), min(5, len(test_ds)), replace=False)
      sampler = torch.utils.data.SubsetRandomSampler(vis_indices)
      vis_dl = torch.utils.data.DataLoader(
          test_ds,
          batch_size=1,
          sampler=sampler,
          num_workers=2,
          collate_fn=collate
      )
      for i, (imgs, targets) in enumerate(vis_dl):
          imgs = imgs.to(device)
          preds = model(imgs)

          visualize_detections(
              imgs[0],
              preds[0]['boxes'],
              preds[0]['labels'],
              preds[0]['scores'],
              targets[0]['boxes'].to(device),
              targets[0]['labels'].to(device),
              vis_indices[i], 
              epoch + 1
          )

    if 'map_per_class' in metrics and 'mar_100_per_class' in metrics:
      print("  Per-class metrics:")

      num_computed_classes = metrics['map_per_class'].shape[0]
      for class_id in range(num_computed_classes):
        if class_id in class_names:
          class_name = class_names[class_id]

          class_map = float(metrics['map_per_class'][class_id])
          class_mar = float(metrics['mar_100_per_class'][class_id])

          if class_map < 0 or class_mar < 0:
              print(f"    {class_name}: No valid metrics available")
              continue

          class_f1 = 2 * (class_map * class_mar) / (class_map + class_mar + 1e-16)
          print(f"    {class_name}: mAP={class_map:.4f}, mAR={class_mar:.4f}, F1={class_f1:.4f}")

    print(f"[RESULTS] Epoch {epoch+1}: ")
    print(f"  Overall mAP = {metrics['map'].item():.4f}")
    print(f"  Overall mAR = {metrics['mar_100'].item():.4f}")
    print(f"  Overall F1 Score = {f1_score.item():.4f}")

    scheduler.step(avg_epoch_loss)

    end_time = time.time()
    print(f"[EPOCH] Time taken: {end_time - start_time:.2f} seconds")

  print("\n[COMPLETE] Training finished!")

if __name__ == "__main__":
    main()