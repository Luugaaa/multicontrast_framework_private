# train_multimodal.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

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
from model import FusionUNet
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


DEVICE = "cuda"

config = {
    "dataset_dir": "datasets/mri_dataset_v2",
    "contrasts_to_use": [1, 5],
    "primary_contrast_id": 1,
    "num_subjects": 200,
    "learning_rate": 3e-5,
    "batch_size": 4,
    "num_epochs": 200,
    "val_split": 0.2,
    "boundary_loss_alpha": 1.0,
    # Misalignment settings
    "apply_misalignment": True, # This is now a core preprocessing step
    "max_rotation": 5,
    "max_translate": 0.05,
    "max_scale": 0.05,
    "single_contrast_epochs": 0,
    "contrast_dropout_rate": 0.3,
    "dark_threshold": 0.1, # Threshold for dark regions in images
    "lambda_features_cosine": 0.4,
    "lambda_features_l1": 3.0,
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

def log_predictions_to_wandb(batch, outputs, epoch, config, processed_image=None, max_dice=None):
    """Logs a comprehensive panel showing misalignment, alignment, and model performance."""
    # Move tensors to CPU and convert to numpy
    for id_ in ["2", "1"] :
        all_contrasts_np = batch[f'contrasts_{id_}'].cpu().numpy()
        
        if id_ == "1" :
            for c_idx in range(all_contrasts_np.shape[1]):
                img = (all_contrasts_np[i, c_idx, 0] * 255).astype(np.uint8)
                panels.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
                panel_captions.append(f"Contrast C{config['contrasts_to_use'][c_idx]}")
            continue
            
        
        aligned_contrasts_t = outputs['warped_moving_contrasts'].cpu()
        aligned_contrasts_np = aligned_contrasts_t.numpy()
        
        primary_contrast_idx = config['contrasts_to_use'].index(config["primary_contrast_id"])
        primary_contrast_np = all_contrasts_np[:, primary_contrast_idx]
        
        gt_masks_raw = batch[f'masks_{id_}'][:, primary_contrast_idx]
        gt_masks_np = gt_masks_raw.cpu().numpy()
        pred_masks_np = (torch.sigmoid(outputs['mask']).cpu().detach().numpy() > 0.5)


        log_images = []
        for i in range(min(all_contrasts_np.shape[0], 4)): # Iterate through batch items
            panels, panel_captions = [], []
            
            # --- Panel 1: Original Misaligned Overlap ---
            h, w = all_contrasts_np.shape[3], all_contrasts_np.shape[4]
            misaligned_overlap_img = np.zeros((h, w, 3), dtype=np.uint8)
            for c_idx in range(all_contrasts_np.shape[1]):
                img = all_contrasts_np[i, c_idx, 0]
                binary_mask = (img > 0.1)
                color = CONTRAST_COLORS[c_idx % len(CONTRAST_COLORS)]
                misaligned_overlap_img[binary_mask] = color

            panels.append(misaligned_overlap_img)
            panel_captions.append("Misaligned Overlap")
            
            # --- Panel 2: Aligned Overlap with Keypoints ---
            aligned_overlap_img = np.zeros((h, w, 3), dtype=np.uint8)
            for c_idx in range(aligned_contrasts_np.shape[1]):
                img = aligned_contrasts_np[i, c_idx, 0]
                binary_mask = (img > 0.1)
                color = CONTRAST_COLORS[c_idx % len(CONTRAST_COLORS)]
                aligned_overlap_img[binary_mask] = color
            
            panels.append(aligned_overlap_img)
            panel_captions.append("Aligned Overlap + Keypoints")
        
            for c_idx in range(aligned_contrasts_np.shape[1]):
                img = (aligned_contrasts_np[i, c_idx, 0] * 255).astype(np.uint8)
                panels.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
                panel_captions.append(f"Aligned Contrast C{config['contrasts_to_use'][c_idx]}")

            
            gt_mask_img = (gt_masks_np[i, 0] * 255).astype(np.uint8)
            panels.append(cv2.cvtColor(gt_mask_img, cv2.COLOR_GRAY2RGB))
            panel_captions.append("GT Mask")

            pred_mask_img = (pred_masks_np[i, 0] * 255).astype(np.uint8)
            panels.append(cv2.cvtColor(pred_mask_img, cv2.COLOR_GRAY2RGB))
            panel_captions.append("Pred Mask")
            
            if processed_image is not None: 
                processed_image_ = (processed_image[i, 0] * 255).astype(np.uint8)
                panels.append(cv2.cvtColor(processed_image_, cv2.COLOR_GRAY2RGB))
                panel_captions.append(f"Contrast merged - max_dice {max_dice:.2f}")
        

            final_image = np.concatenate(panels, axis=1)
            final_caption = f"Subj {batch['subject_id'][i]}: " + " | ".join(panel_captions)
            log_images.append(wandb.Image(final_image, caption=final_caption))
            

    wandb.log({"Validation Predictions": log_images, "epoch": epoch})

# --- 3. Main Training Function ---
def main():
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    wandb.init(project="multimodal_agnostic_unet_prototype_3", config=config)
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
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=8,      # Start with 4 or 8 and see what works best
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    model = FusionUNet(
        in_channels=1, 
        out_channels=1, 
    ).to(DEVICE)
    
    
    loss_fn = BoundaryLoss(alpha=config["boundary_loss_alpha"])
    l1_loss_fn = nn.L1Loss()
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    lambda_features_cosine = config['lambda_features_cosine']
    lambda_features_l1 = config['lambda_features_l1']
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    print("🚀 Starting training...")
    print(f"   - Phase 1 (Epochs 1-{config['single_contrast_epochs']}): Training with a single random contrast.")
    print(f"   - Phase 2 (Epochs >{config['single_contrast_epochs']}): Training with all contrasts.")

    best_val_dice = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss, total_mask_loss, total_boundary_loss, total_epoch_align_loss, total_dff_loss = 0, 0, 0, 0, 0
        total_features_loss = 0
        total_features_loss_l1 = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            features_list = []
            ids = ["1", "2"]
            loss = 0.0
            
            for id_ in ids : 
                all_contrasts = batch[f'contrasts_{id_}'].to(DEVICE)
                all_masks = batch[f'masks_{id_}'].to(DEVICE)
                

                contrasts_to_use = config['contrasts_to_use']
                ref_contrast = random.choice(contrasts_to_use) ## we choose a random ref contrast during training !
                ref_idx = contrasts_to_use.index(ref_contrast)
                
                reference_contrast = all_contrasts[:, ref_idx]
                reference_mask = all_masks[:, ref_idx].contiguous()

                moving_contrasts = [ 
                    all_contrasts[:, i] for i in range(len(contrasts_to_use)) if i != ref_idx 
                ]

                
                boundaries = masks_to_boundaries_gpu(reference_mask)
                
                outputs = model(reference_contrast, moving_contrasts)
                features_list.append(outputs["features"]) # B, N, C, H, W
                
  
                seg_loss, mask_loss, boundary_loss = loss_fn(outputs, {"mask": reference_mask, "boundary": boundaries})
                
                loss += seg_loss 

                if id_ == ids[-1] :
                    features_1 = features_list[0]
                    features_2 = features_list[1]
                    for feat_1, feat_2 in zip(features_1, features_2):
                        features_cosine_loss = (1 - cosine_sim(feat_1.flatten(2), feat_2.flatten(2)).mean()) * lambda_features_cosine
                        features_loss_l1 = l1_loss_fn(feat_1.flatten(2), feat_2.flatten(2)) * lambda_features_l1
                        loss += features_cosine_loss + features_loss_l1
                        
                        total_features_loss += features_cosine_loss.item() 
                        total_features_loss_l1 += features_loss_l1.item() 
                
                total_loss += loss.item()
                total_mask_loss += mask_loss.item()
                total_boundary_loss += boundary_loss.item()  
            
            
            loss.backward()
                          
            optimizer.step()

        wandb.log({"train_losses/total_loss": total_loss, 
                   "train_losses/mask_loss": total_mask_loss, 
                   "train_losses/boundary_loss": total_boundary_loss, 
                   "train_losses/features_loss": total_features_loss,
                   "train_losses/features_loss_l1": total_features_loss_l1
        })
            
        
        model.eval()
        total_val_dice = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
                ids = ["1", "2"]
                for id_ in ids :
                    all_contrasts = batch[f'contrasts_{id_}'].to(DEVICE)
                    all_masks = batch[f'masks_{id_}'].to(DEVICE)

                    contrasts_to_use = config['contrasts_to_use']
                    ref_idx = contrasts_to_use.index(config['primary_contrast_id'])
                    
                    reference_contrast = all_contrasts[:, ref_idx]
                    reference_mask = all_masks[:, ref_idx].contiguous()
                    
                    moving_contrasts = [
                        all_contrasts[:, i] for i in range(len(contrasts_to_use)) if i!= ref_idx
                    ]

                    outputs = model(reference_contrast, moving_contrasts)
                    total_val_dice += dice_metric(outputs['mask'], reference_mask)/len(ids)
                
                if i == 0:
                    with torch.no_grad():
                        warped_tensors = outputs['warped_moving_contrasts'][:, 1, :, :, :].squeeze(2) # B, C, 1, SIZE, SIZE
                        binary_reference_contrast = ((reference_contrast) >= 0.1) & ((reference_contrast) <= 0.35)
                        binary_moving_contrasts = ((warped_tensors) >= 0.1) & ((warped_tensors) <= 0.35)

                        processed_image = (binary_reference_contrast + binary_moving_contrasts)/2
                        max_absolute_dice = dice_metric(processed_image, reference_mask)
                
            
            
                    log_predictions_to_wandb(batch, outputs, epoch, config, processed_image.cpu().numpy(), max_dice=max_absolute_dice)
        
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
