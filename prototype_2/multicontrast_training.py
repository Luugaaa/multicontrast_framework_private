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

import torch
import torch.nn as nn

class CenterOfMassLoss(nn.Module):
    """
    Calculates the squared Euclidean distance between the centers of mass of two images.
    Expects inputs to be probability maps or non-negative feature maps.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _calculate_com(self, image):
        """Calculates the center of mass for a batch of images."""
        B, C, H, W = image.shape
        image = F.relu(image - config["dark_threshold"])
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            [torch.arange(H, device=image.device, dtype=torch.float32),
             torch.arange(W, device=image.device, dtype=torch.float32)],
            indexing='ij'
        ) # Shapes (H, W)

        # Total mass for each image in the batch
        total_mass = image.sum(dim=[1, 2, 3]) + self.eps # Shape (B,)

        # Calculate weighted coordinates
        weighted_y = (y_coords * image).sum(dim=[1, 2, 3]) # Shape (B,)
        weighted_x = (x_coords * image).sum(dim=[1, 2, 3]) # Shape (B,)

        # Calculate center of mass coordinates
        com_y = weighted_y / total_mass
        com_x = weighted_x / total_mass

        return torch.stack([com_x, com_y], dim=1) # Shape (B, 2)

    def forward(self, ref_image, warped_image):
        """
        Args:
            ref_image (torch.Tensor): The reference image (B, C, H, W).
            warped_image (torch.Tensor): The warped moving image (B, C, H, W).
        """
        com_ref = self._calculate_com(ref_image)
        com_warped = self._calculate_com(warped_image)

        # Calculate the squared Euclidean distance between the CoMs
        loss = torch.mean((com_ref - com_warped)**2)
        
        return loss
    
class MassQuantileLoss(nn.Module):
    """
    Calculates the squared distance between intensity-weighted quantile points of two images.
    """
    def __init__(self, quantiles=[0.25, 0.75], eps=1e-8):
        super().__init__()
        self.quantiles = quantiles
        self.eps = eps

    def _calculate_quantiles(self, image):
        """Calculates the quantile coordinates for a batch of images."""
        image = F.relu(image - config["dark_threshold"])
        B, C, H, W = image.shape
        
        # Total mass for normalization
        total_mass = image.sum(dim=[1, 2, 3]) + self.eps

        # --- Y-axis Quantiles ---
        mass_y = image.sum(dim=3) # Project mass onto Y-axis, Shape (B, C, H)
        cum_mass_y = torch.cumsum(mass_y, dim=2) # Cumulative sum along Y
        
        # --- X-axis Quantiles ---
        mass_x = image.sum(dim=2) # Project mass onto X-axis, Shape (B, C, W)
        cum_mass_x = torch.cumsum(mass_x, dim=2)

        quantile_points = []
        for q in self.quantiles:
            # Find target cumulative mass for this quantile
            target_mass = q * total_mass.view(B, 1, 1)

            # Find the index where the cumulative mass exceeds the target
            # searchsorted is efficient for finding this insertion point
            q_y = torch.searchsorted(cum_mass_y.contiguous(), target_mass).float().squeeze(-1)
            q_x = torch.searchsorted(cum_mass_x.contiguous(), target_mass).float().squeeze(-1)
            
            # We get one point per channel, let's average them
            quantile_points.append(torch.stack([q_x.mean(dim=1), q_y.mean(dim=1)], dim=1))

        # Returns a list of tensors, each of shape (B, 2)
        return quantile_points

    def forward(self, ref_image, warped_image):
        """
        Args:
            ref_image (torch.Tensor): The reference image (B, C, H, W).
            warped_image (torch.Tensor): The warped moving image (B, C, H, W).
        """
        quantile_points_ref = self._calculate_quantiles(ref_image)
        quantile_points_warped = self._calculate_quantiles(warped_image)

        total_loss = 0.0
        # Calculate the distance for each pair of corresponding quantile points
        for q_ref, q_warped in zip(quantile_points_ref, quantile_points_warped):
            q_ref /= torch.tensor([ref_image.shape[3], ref_image.shape[2]], device=ref_image.device)
            q_warped /= torch.tensor([ref_image.shape[3], ref_image.shape[2]], device=ref_image.device)
            total_loss += torch.mean(((q_ref - q_warped)*20)**4) ## penalize a lot when the distance is large  only
        
        return total_loss
    
    
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
    "batch_size": 2,
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
def draw_keypoints(image, com_coords, quantile_coords_list, color, size_factor=1):
    """Draws Center of Mass and Quantile points on an image."""
    # Draw Center of Mass (larger circle)
    if com_coords : 
        com_x, com_y = int(com_coords[0]), int(com_coords[1])
        cv2.circle(image, (com_x, com_y), radius=int(3*size_factor), color=color, thickness=-1)
        cv2.circle(image, (com_x, com_y), radius=int(3*size_factor), color=(255, 255, 255), thickness=1)
    
    # Draw Quantile Points (smaller circles)
    if quantile_coords_list : 
        for q_coords in quantile_coords_list:
            q_x, q_y = int(q_coords[0]), int(q_coords[1])
            cv2.circle(image, (q_x, q_y), radius=int(2*size_factor), color=color, thickness=-1)
            cv2.circle(image, (q_x, q_y), radius=int(2*size_factor), color=(255, 255, 255), thickness=1)

def log_predictions_to_wandb(batch, outputs, epoch, config, processed_image=None, max_dice=None):
    """Logs a comprehensive panel showing misalignment, alignment, and model performance."""
    # Move tensors to CPU and convert to numpy
    all_contrasts_np = batch['contrasts'].cpu().numpy()
    aligned_contrasts_t = outputs['warped_moving_contrasts'].cpu()
    aligned_contrasts_np = aligned_contrasts_t.numpy()
    
    primary_contrast_idx = config['contrasts_to_use'].index(config["primary_contrast_id"])
    primary_contrast_np = all_contrasts_np[:, primary_contrast_idx]
    
    gt_masks_raw = batch['masks'][:, primary_contrast_idx]
    gt_masks_np = gt_masks_raw.cpu().numpy()
    pred_masks_np = (torch.sigmoid(outputs['mask']).cpu().detach().numpy() > 0.5)

    # --- Instantiate helpers for keypoint calculation ---
    # com_calculator = CenterOfMassLoss()._calculate_com
    quantile_calculator = MassQuantileLoss(quantiles=[0.25, 0.5, 0.75])._calculate_quantiles

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
        
        with torch.no_grad():
            for c_idx in range(all_contrasts_np.shape[1]):
                # Get the single image tensor for calculation
                img_tensor = torch.tensor(all_contrasts_np[i, c_idx]).unsqueeze(0)
                # prob_map = torch.sigmoid(img_tensor)
                color = CONTRAST_COLORS[c_idx % len(CONTRAST_COLORS)]

                # Calculate keypoints
                # com = com_calculator(img_tensor)[0].cpu().numpy()
                quantiles = [q[0].cpu().numpy() for q in quantile_calculator(img_tensor)]

                if c_idx>0 : 
                    size_factor = 1.5
                else : size_factor = 1.0
                color_ = [int(el//2) for el in color]
                draw_keypoints(misaligned_overlap_img, None, quantiles, color_, size_factor)


        panels.append(misaligned_overlap_img)
        panel_captions.append("Misaligned Overlap")
        
        # --- Panel 2: Aligned Overlap with Keypoints ---
        aligned_overlap_img = np.zeros((h, w, 3), dtype=np.uint8)
        for c_idx in range(aligned_contrasts_np.shape[1]):
            img = aligned_contrasts_np[i, c_idx, 0]
            binary_mask = (img > 0.1)
            color = CONTRAST_COLORS[c_idx % len(CONTRAST_COLORS)]
            aligned_overlap_img[binary_mask] = color
        
        # Calculate and draw keypoints on the aligned overlap image
        with torch.no_grad():
            for c_idx in range(aligned_contrasts_t.shape[1]):
                # Get the single image tensor for calculation
                img_tensor = aligned_contrasts_t[i, c_idx].unsqueeze(0)
                # prob_map = torch.sigmoid(img_tensor)
                color = CONTRAST_COLORS[c_idx % len(CONTRAST_COLORS)]

                # Calculate keypoints
                # com = com_calculator(img_tensor)[0].cpu().numpy()
                quantiles = [q[0].cpu().numpy() for q in quantile_calculator(img_tensor)]

                if c_idx>0 : 
                    size_factor = 1.5
                else : size_factor = 1.0
                color_ = [int(el//2) for el in color]
                draw_keypoints(aligned_overlap_img, None, quantiles, color_, size_factor)

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
            # log_images.append(wandb.Image(processed_image, caption=f"Processed Image for Subj {batch['subject_id'][i]}, max_dice:{max_dice}"))
    

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
    alignment_loss_fn = DiceLoss()
    dff_loss_fn = BendingEnergyLoss()
    # com_loss_fn = CenterOfMassLoss()
    quantile_loss_fn = MassQuantileLoss(quantiles=[0.25, 0.5, 0.75]) # Using quartiles + median


    lambda_alignment = 3.0
    lambda_dff = 10.0
    lambda_reg = 10.0
    lambda_com = 0.007
    lambda_quantile = 0.002
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    print("🚀 Starting training...")
    print(f"   - Phase 1 (Epochs 1-{config['single_contrast_epochs']}): Training with a single random contrast.")
    print(f"   - Phase 2 (Epochs >{config['single_contrast_epochs']}): Training with all contrasts.")

    best_val_dice = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss, total_mask_loss, total_boundary_loss, total_epoch_align_loss, total_dff_loss = 0, 0, 0, 0, 0
        total_reg_loss = 0
        total_epoch_com_loss = 0
        total_epoch_quantile_loss = 0
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
            warped_contrasts = outputs["warped_moving_contrasts"] 

            num_warped = warped_contrasts.size()[1] - 1 # Exclude the reference image
            
            tmp_total_com_loss = 0.0
            tmp_total_quantile_loss = 0.0
            if num_warped > 0:
                for i in range(num_warped):
                    # The first image in the stack is the reference
                    warped_img = warped_contrasts[:, i+1, ...]
                    
                    # It's best to compare the predicted masks if available, or the images themselves
                    # For example, using sigmoid to get probability maps
                    # ref_prob = torch.sigmoid(reference_contrast)
                    # warped_prob = torch.sigmoid(warped_img)

                    # tmp_total_com_loss += com_loss_fn(reference_contrast, warped_img)
                    tmp_total_quantile_loss += quantile_loss_fn(reference_contrast, warped_img)

                # com_loss = tmp_total_com_loss / num_warped
                quantile_loss = tmp_total_quantile_loss / num_warped
            else:
                # com_loss = 0.0
                quantile_loss = 0.0
                
            seg_loss, mask_loss, boundary_loss = loss_fn(outputs, {"mask": reference_mask, "boundary": boundaries})
            
            total_align_loss = 0
            if moving_contrasts and 'warped_moving_contrasts' in outputs:
                warped_tensors = outputs['warped_moving_contrasts'] # B, C, 1, SIZE, SIZE
                adapted_warped_tensors = warped_tensors.transpose(0, 1)
                for i, warped_mov_contrast in enumerate(adapted_warped_tensors):
                    if i==0 : continue #skip the ref contrast
                    total_align_loss += alignment_loss_fn(warped_mov_contrast.contiguous(), reference_contrast.contiguous())
            
            # dff_loss = dff_loss_fn(outputs["predicted_ddfs"])
            # Combined Loss
            deviations = outputs["predicted_deviations"]
            reg_loss = torch.mean(deviations**2) 
            
            loss = (seg_loss 
                + lambda_alignment * total_align_loss 
                + lambda_reg * reg_loss
                # + lambda_com * com_loss
                + lambda_quantile * quantile_loss)
            
            
            
            loss.backward()
            optimizer.step()
            
            
            total_loss += loss.item()
            total_mask_loss += mask_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_epoch_align_loss += lambda_alignment * total_align_loss.item()
            total_reg_loss += reg_loss.item() * lambda_reg
            # total_epoch_com_loss += com_loss.item() * lambda_com
            total_epoch_quantile_loss += quantile_loss.item() * lambda_quantile
            # total_dff_loss += dff_loss.item() * lambda_dff

        wandb.log({"train_losses/total_loss": total_loss, 
                   "train_losses/mask_loss": total_mask_loss, 
                   "train_losses/boundary_loss": total_boundary_loss, 
                   "train_losses/align_loss": total_epoch_align_loss, 
                   "train_losses/dff_loss": total_dff_loss, 
                   "train_losses/reg_loss": total_reg_loss,
                #    "train_losses/com_loss": total_epoch_com_loss,
                    # "metrics/max_abosulte_dice": max_absolute_dice,
                   "train_losses/quantile_loss": total_epoch_quantile_loss})
            
        
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
                    with torch.no_grad():
                        warped_tensors = outputs['warped_moving_contrasts'][:, 1, :, :, :].squeeze(2) # B, C, 1, SIZE, SIZE
                        binary_reference_contrast = ((reference_contrast) >= 0.1) & ((reference_contrast) <= 0.4)
                        binary_moving_contrasts = ((warped_tensors) >= 0.1) & ((warped_tensors) <= 0.4)
                        # sum_image = (reference_contrast + moving_contrasts[0])/2
                        processed_image = (binary_reference_contrast + binary_moving_contrasts)/2
                        # print((reference_contrast), "\n\n\n\n", (moving_contrasts[0]), "\n\n\n\n",processed_image)
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
