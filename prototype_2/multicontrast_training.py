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
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

# --- Import the NEW model ---
from model import AlignmentAndFusionUNet
from data_loader import MultiContrastDataset
from train_unet import DiceLoss, dice_metric


def dice_metric(logits, targets):
    loss = DiceLoss()
    return 1 - loss(logits, targets).item()

def masks_to_boundaries_gpu(masks, kernel_size=3):
    with torch.no_grad():
        eroded_masks = -F.max_pool2d(-masks.float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return F.relu(masks.float() - eroded_masks)

class BoundaryLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
    def forward(self, outputs, targets):
        mask_logits, boundary_logits = outputs["mask"], outputs["boundary"]
        gt_mask, gt_boundary = targets["mask"], targets["boundary"]
        loss_mask = self.dice_loss(mask_logits, gt_mask)
        loss_boundary = self.dice_loss(boundary_logits, gt_boundary)
        return loss_mask + self.alpha * loss_boundary, loss_mask, loss_boundary

class BendingEnergyLoss(nn.Module):
    """
    Calculates the bending energy of a dense displacement field to enforce smoothness.
    """
    def forward(self, ddf):
        # ddf shape: (B, 2, H, W)
        dy = ddf[:, :, 1:, :] - ddf[:, :, :-1, :]
        dx = ddf[:, :, :, 1:] - ddf[:, :, :, :-1]
        return torch.mean(dx**2) + torch.mean(dy**2)
    
# # --- 1. Configuration ---
# if torch.cuda.is_available():
#     DEVICE = "cuda"
# else:
#     DEVICE = "cpu"

DEVICE = "mps"

config = {
    "dataset_dir": "datasets/mri_dataset_v2",
    "contrasts_to_use": [1, 5],
    "primary_contrast_id": 1,
    "num_subjects": 200,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "num_epochs": 200,
    "val_split": 0.2,
    "boundary_loss_alpha": 1.0,
    # Misalignment settings
    "apply_misalignment": True, # This is now a core preprocessing step
    "max_rotation": 10,
    "max_translate": 0.08,
    "max_scale": 0.15,
    "single_contrast_epochs": 0,
    "contrast_dropout_rate": 0.3
}

# --- 2. Visualization Helper ---
CONTRAST_COLORS = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
    (255, 0, 255), # Magenta
]

def log_predictions_to_wandb(batch, outputs, epoch, config):
    """Logs a comprehensive panel showing misalignment and model performance."""
    all_contrasts = batch['contrasts'].cpu().numpy()
    primary_contrast_el_id = config['contrasts_to_use'].index(config["primary_contrast_id"])
    primary_contrast = batch['contrasts'][:, primary_contrast_el_id].cpu().numpy()
    aligned_contrasts = outputs['warped_moving_contrasts'].cpu().numpy()
    gt_masks_raw = batch['masks'][:, primary_contrast_el_id]
    gt_masks = gt_masks_raw.cpu().numpy()
    
    pred_masks = (torch.sigmoid(outputs['mask']).cpu().detach().numpy() > 0.5)
    gt_boundaries = masks_to_boundaries_gpu(gt_masks_raw.cpu()).numpy()
    pred_boundaries = (torch.sigmoid(outputs['boundary']).cpu().detach().numpy() > 0.5)

    log_images = []
    for i in range(min(all_contrasts.shape[0], 4)):
        panels, panel_captions = [], []
        
        # Panel 1: Color overlap of all misaligned contrasts
        h, w = all_contrasts.shape[3], all_contrasts.shape[4]
        overlap_img = np.zeros((h, w, 3), dtype=np.uint8)
        for c_idx in range(all_contrasts.shape[1]):
            img = all_contrasts[i, c_idx, 0]
            binary_mask = (img > 0.1)
            color = CONTRAST_COLORS[c_idx % len(CONTRAST_COLORS)]
            overlap_img[binary_mask] = color
        panels.append(overlap_img)
        panel_captions.append("Misalign Overlap")
        
        # Panel 1 bis: Color overlap of all misaligned contrasts
        h, w = aligned_contrasts.shape[3], aligned_contrasts.shape[4]
        overlap_img = np.zeros((h, w, 3), dtype=np.uint8)
        for c_idx in range(aligned_contrasts.shape[1]):
            img = aligned_contrasts[i, c_idx, 0]
            binary_mask = (img > 0.1)
            color = CONTRAST_COLORS[c_idx % len(CONTRAST_COLORS)]
            overlap_img[binary_mask] = color
        panels.append(overlap_img)
        panel_captions.append("Misalign Overlap after spatial transformer")
        

        # Panel 2: The primary contrast (decoder's spatial reference)
        primary_img_np = (primary_contrast[i, 0] * 255).astype(np.uint8)
        panels.append(cv2.cvtColor(primary_img_np, cv2.COLOR_GRAY2RGB))
        panel_captions.append(f"Ref: C{config['primary_contrast_id']}")
        
        # Panel 3 & 4: GT and Predicted Mask
        gt_mask_img = (gt_masks[i, 0] * 255).astype(np.uint8)
        panels.append(cv2.cvtColor(gt_mask_img, cv2.COLOR_GRAY2RGB))
        panel_captions.append("GT Mask")
        pred_mask_img = (pred_masks[i, 0] * 255).astype(np.uint8)
        panels.append(cv2.cvtColor(pred_mask_img, cv2.COLOR_GRAY2RGB))
        panel_captions.append("Pred Mask")

        # Panel 5 & 6: GT and Predicted Boundaries
        gt_boundary_img = (gt_boundaries[i, 0] * 255).astype(np.uint8)
        panels.append(cv2.cvtColor(gt_boundary_img, cv2.COLOR_GRAY2RGB))
        panel_captions.append("GT Boundary")
        pred_boundary_img = (pred_boundaries[i, 0] * 255).astype(np.uint8)
        panels.append(cv2.cvtColor(pred_boundary_img, cv2.COLOR_GRAY2RGB))
        panel_captions.append("Pred Boundary")

        final_image = np.concatenate(panels, axis=1)
        final_caption = f"Subj {batch['subject_id'][i]}: " + " | ".join(panel_captions)
        log_images.append(wandb.Image(final_image, caption=final_caption))
    
    wandb.log({"Validation Predictions": log_images, "epoch": epoch})

# --- 3. Main Training Function ---
def main():
    wandb.init(project="multimodal_agnostic_unet_prototype_2", config=config)
    print(f"✅ Using device: {DEVICE}")

    subject_ids = list(range(config["num_subjects"]))
    train_ids, val_ids = train_test_split(subject_ids, test_size=config["val_split"], random_state=42)

    dataset_args = {
        "dataset_dir": config["dataset_dir"],
        "contrasts_to_use": config["contrasts_to_use"],
        "primary_contrast_id": config["primary_contrast_id"],
        "apply_misalignment": config["apply_misalignment"],
        "max_rotation": config["max_rotation"],
        "max_translate": config["max_translate"],
        "max_scale": config["max_scale"]
    }
    
    train_dataset = MultiContrastDataset(subject_ids=train_ids, **dataset_args)
    val_dataset = MultiContrastDataset(subject_ids=val_ids, **dataset_args)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = AlignmentAndFusionUNet(
        in_channels=1, 
        out_channels=1, 
    ).to(DEVICE)
    
    
    loss_fn = BoundaryLoss(alpha=config["boundary_loss_alpha"])
    alignment_loss_fn = nn.MSELoss()
    dff_loss_fn = BendingEnergyLoss()
    lambda_alignment = 30.0
    lambda_dff = 10.0
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    print("🚀 Starting training...")
    print(f"   - Phase 1 (Epochs 1-{config['single_contrast_epochs']}): Training with a single random contrast.")
    print(f"   - Phase 2 (Epochs >{config['single_contrast_epochs']}): Training with all contrasts.")

    best_val_dice = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss, total_mask_loss, total_boundary_loss, total_epoch_align_loss, total_dff_loss = 0, 0, 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            all_contrasts = batch['contrasts'].to(DEVICE)
            all_masks = batch['masks'].to(DEVICE)
            

            # Use integer indexing to remove the singleton dimension, creating a 4D tensor
            if epoch < config["single_contrast_epochs"]:
                random_contrast_idx = config['contrasts_to_use'].index(random.choice(config['contrasts_to_use']))
                
                reference_contrast = all_contrasts[:, random_contrast_idx]
                reference_mask = all_masks[:, random_contrast_idx].contiguous()
                moving_contrasts = []
                
            else:
                contrasts_to_use = config['contrasts_to_use']
                ref_contrast = random.choice(contrasts_to_use) ## we choose a random ref contrast during training !
                ref_idx = contrasts_to_use.index(ref_contrast)
                
                reference_contrast = all_contrasts[:, ref_idx]
                reference_mask = all_masks[:, ref_idx].contiguous()

                moving_contrasts = [ 
                    all_contrasts[:, i] for i in range(len(contrasts_to_use)) if i != ref_idx 
                ]

            
            boundaries = masks_to_boundaries_gpu(reference_mask)
            
            optimizer.zero_grad()
            outputs = model(reference_contrast, moving_contrasts)
            
            seg_loss, mask_loss, boundary_loss = loss_fn(outputs, {"mask": reference_mask, "boundary": boundaries})
            
            total_align_loss = 0
            if moving_contrasts and 'warped_moving_contrasts' in outputs:
                warped_tensors = outputs['warped_moving_contrasts'] # B, C, 1, SIZE, SIZE
                adapted_warped_tensors = warped_tensors.transpose(0, 1)
                for i, warped_mov_contrast in enumerate(adapted_warped_tensors):
                    if i==0 : continue #skip the ref contrast
                    total_align_loss += alignment_loss_fn(warped_mov_contrast, reference_contrast)
            
            dff_loss = dff_loss_fn(outputs["predicted_ddfs"])
            # Combined Loss
            loss = seg_loss + lambda_alignment * total_align_loss + dff_loss * lambda_dff
    
    
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mask_loss += mask_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_epoch_align_loss += lambda_alignment * total_align_loss.item()
            total_dff_loss += dff_loss.item() * lambda_dff
            
        wandb.log({"train_losses/total_loss": total_loss, "train_losses/mask_loss": total_mask_loss, "train_losses/boundary_loss": total_boundary_loss, "train_losses/align_loss": total_epoch_align_loss, "train_losses/dff_loss": total_dff_loss})
            
        
        model.eval()
        total_val_dice = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
                if epoch < config["single_contrast_epochs"]:
                    random_contrast_idx = config['contrasts_to_use'].index(random.choice(config['contrasts_to_use']))
                
                    reference_contrast = all_contrasts[:, random_contrast_idx]
                    reference_mask = all_masks[:, random_contrast_idx].contiguous()
                    moving_contrasts = []
                else : 
                    # for k in range(len(config["contrasts_to_use"])):
                        # if random.random() <= config["contrast_dropout_rate"]:
                            
                    all_contrasts = batch['contrasts'].to(DEVICE)
                    all_masks = batch['masks'].to(DEVICE)

                    contrasts_to_use = config['contrasts_to_use']
                    ref_idx = contrasts_to_use.index(config['primary_contrast_id'])
                    
                    reference_contrast = all_contrasts[:, ref_idx]
                    reference_mask = all_masks[:, ref_idx].contiguous()
                    
                    moving_contrasts = [
                        all_contrasts[:, i] for i in range(len(contrasts_to_use)) if i!= ref_idx
                    ]

                outputs = model(reference_contrast, moving_contrasts)
                total_val_dice += dice_metric(outputs['mask'], reference_mask)
                
                if i == 0:
                    log_predictions_to_wandb(batch, outputs, epoch, config)
        
        scheduler.step()
        
        avg_val_dice = total_val_dice / len(val_loader)
        wandb.log({"epoch": epoch, "val_dice_mask": avg_val_dice})
        print(f"Epoch {epoch+1}: Val Dice (Mask): {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "multicontrast_model.pth")
            print(f"🎉 New best model saved with Val Dice: {best_val_dice:.4f}")

    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()
