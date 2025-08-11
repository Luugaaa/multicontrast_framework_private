# train_multimodal.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import numpy as np
import cv2

# --- Import the NEW model ---
from model import BoundaryAwareUNet
from data_loader import MultiContrastDataset
from train_unet import DiceLoss, dice_metric

# --- 1. Configuration (with new alpha parameter) ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

config = {
    "dataset_dir": "datasets/mri_dataset_v2",
    "contrasts_to_use": [1, 5],
    "primary_contrast_for_vis": 1,
    "num_subjects": 200,
    "learning_rate": 1e-4,
    "batch_size": 2,
    "num_epochs": 100,
    "val_split": 0.2,
    "boundary_loss_alpha": 1.0, # NEW: Weight for the boundary loss component
}

# --- NEW: Boundary Generation Helper ---
# This function efficiently creates boundary maps from masks on the GPU.
def masks_to_boundaries_gpu(masks, kernel_size=3):
    """
    Uses morphological erosion on the GPU to quickly generate boundary maps.
    """
    with torch.no_grad():
        # Invert mask for max_pool to act as erosion
        eroded_masks = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        boundaries = F.relu(masks - eroded_masks)
    return boundaries

# --- NEW: Composite Loss Function ---
class BoundaryLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()

    def forward(self, outputs, targets):
        # Unpack predictions and ground truths
        mask_logits = outputs["mask"]
        boundary_logits = outputs["boundary"]
        gt_mask = targets["mask"]
        gt_boundary = targets["boundary"]

        # Calculate individual losses
        loss_mask = self.dice_loss(mask_logits, gt_mask)
        loss_boundary = self.dice_loss(boundary_logits, gt_boundary)
        
        # Combine losses
        total_loss = loss_mask + self.alpha * loss_boundary
        
        return total_loss, loss_mask, loss_boundary

# --- 2. W&B Visualization Helper (Updated for Boundary Visualization) ---
def log_predictions_to_wandb(batch, outputs, epoch):
    contrasts = batch['contrasts'].cpu().numpy()
    gt_masks = batch['mask'].cpu().numpy()
    gt_boundaries = masks_to_boundaries_gpu(batch['mask']).cpu().numpy()
    
    primary_idx_in_list = config['contrasts_to_use'].index(config['primary_contrast_for_vis'])
    
    pred_masks = (torch.sigmoid(outputs['mask']).cpu().detach().numpy() > 0.5).astype(np.float32)
    pred_boundaries = (torch.sigmoid(outputs['boundary']).cpu().detach().numpy() > 0.5).astype(np.float32)

    log_images = []
    for i in range(min(contrasts.shape[0], 4)):
        primary_img = contrasts[i, primary_idx_in_list, 0]
        primary_img = (primary_img - primary_img.min()) / (primary_img.max() - primary_img.min() + 1e-6)
        
        # Create a 5-panel comparison image
        comparison_img = np.concatenate([
            primary_img, 
            gt_masks[i, 0], 
            pred_masks[i, 0],
            gt_boundaries[i, 0],
            pred_boundaries[i, 0]], axis=1) * 255
        
        log_images.append(wandb.Image(
            comparison_img.astype(np.uint8),
            caption=f"Subject {batch['subject_id'][i]}: Input | GT Mask | Pred Mask | GT Boundary | Pred Boundary"
        ))
    wandb.log({"predictions": log_images, "epoch": epoch})

# --- 3. Main Training Function (Updated) ---
def main():
    wandb.init(project="multimodal_agnostic_unet", config=config)
    print(f"✅ Using device: {DEVICE}")

    # --- Data Loading (remains the same) ---
    subject_ids = list(range(config["num_subjects"]))
    train_ids, val_ids = train_test_split(subject_ids, test_size=config["val_split"], random_state=42)
    train_dataset = MultiContrastDataset(subject_ids=train_ids, dataset_dir=config["dataset_dir"], contrasts_to_use=config["contrasts_to_use"], primary_contrast_idx=config["primary_contrast_for_vis"])
    val_dataset = MultiContrastDataset(subject_ids=val_ids, dataset_dir=config["dataset_dir"], contrasts_to_use=config["contrasts_to_use"], primary_contrast_idx=config["primary_contrast_for_vis"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # --- Model Setup ---
    print("Setting up BoundaryAwareUNet model...")
    model = BoundaryAwareUNet(in_channels=1, out_channels=1).to(DEVICE)
    
    # --- Training Setup ---
    loss_fn = BoundaryLoss(alpha=config["boundary_loss_alpha"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print("🚀 Starting training...")
    best_val_dice = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss, total_mask_loss, total_boundary_loss = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            contrasts, masks = batch['contrasts'].to(DEVICE), batch['mask'].to(DEVICE)
            # NEW: Generate boundary targets on-the-fly
            boundaries = masks_to_boundaries_gpu(masks)
            
            optimizer.zero_grad()
            outputs = model(contrasts)
            
            loss, mask_loss, boundary_loss = loss_fn(outputs, {"mask": masks, "boundary": boundaries})
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mask_loss += mask_loss.item()
            total_boundary_loss += boundary_loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_mask_loss = total_mask_loss / len(train_loader)
        avg_boundary_loss = total_boundary_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        total_val_dice = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
                contrasts, masks = batch['contrasts'].to(DEVICE), batch['mask'].to(DEVICE)
                outputs = model(contrasts)
                # We still evaluate performance based on the primary task: mask segmentation
                total_val_dice += dice_metric(outputs['mask'], masks)
                if i == 0:
                    log_predictions_to_wandb(batch, outputs, epoch)

        avg_val_dice = total_val_dice / len(val_loader)
        
        wandb.log({
            "epoch": epoch,
            "total_train_loss": avg_train_loss,
            "train_mask_loss": avg_mask_loss,
            "train_boundary_loss": avg_boundary_loss,
            "val_dice_mask": avg_val_dice
        })
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Dice (Mask): {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "boundary_aware_model.pth")
            print(f"🎉 New best model saved with Val Dice: {best_val_dice:.4f}")

    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()