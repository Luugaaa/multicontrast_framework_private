import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# --- 1. Configuration & Hyperparameters ---
DATASET_DIR = "datasets/mri_dataset"
MODEL_SAVE_PATH = "unet_lesion_segmentation.pth"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

config = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 50,
    "img_size": 64,
    "val_split": 0.2,
    "architecture": "U-Net",
    "num_subjects": 200 # Total unique images per contrast
}

# --- 2. U-Net Model Architecture (Unchanged) ---
class SlimDoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2 with fewer channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """A U-Net with a reduced number of channels at each layer."""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        # Halving the number of channels: 32, 64, 128, 256
        self.inc = SlimDoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), SlimDoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), SlimDoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), SlimDoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = SlimDoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = SlimDoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = SlimDoubleConv(64, 32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv3(x)
        return self.outc(x)


# --- 3. Custom Dataset (Unchanged) ---
class MRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = (image.astype(np.float32) / 255.0)[np.newaxis, :, :]
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask.astype(np.float32) / 255.0)[np.newaxis, :, :]
        return torch.from_numpy(image), torch.from_numpy(mask)

# --- 4. Metrics, Loss, Visualization (Unchanged) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        preds = torch.sigmoid(logits); preds = preds.view(-1); targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff

def dice_metric(logits, targets, smooth=1e-6):
    preds = torch.sigmoid(logits); preds = (preds > 0.5).float(); preds = preds.view(-1); targets = targets.view(-1)
    intersection = (preds * targets).sum()
    score = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return score.item()

def log_predictions_to_wandb(images, masks, outputs, epoch, contrast_name):
    images_np = images.cpu().numpy(); masks_np = masks.cpu().numpy()
    preds_np = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(np.float32)
    log_images = []
    for i in range(min(images_np.shape[0], 8)):
        comparison_img = np.concatenate([images_np[i, 0], masks_np[i, 0], preds_np[i, 0]], axis=1) * 255
        log_images.append(wandb.Image(comparison_img, caption="Input | Ground Truth | Prediction"))
    wandb.log({f"predictions_{contrast_name}": log_images, "epoch": epoch})

# --- 5. Main Training and Validation ---
def main():
    wandb.init(project="multimodal_prototype_1", config=config)
    print(f"✅ Using device: {DEVICE}")

    # --- Data Loading and Splitting (CORRECTED to prevent data leakage) ---
    subject_ids = list(range(config["num_subjects"]))
    train_ids, val_ids = train_test_split(subject_ids, test_size=config["val_split"], random_state=42)

    train_imgs, train_msks = [], []
    for subject_id in train_ids:
        for contrast_idx in [1, 2, 3]:
            filename = f"{subject_id:04d}.png"
            img_path = os.path.join(DATASET_DIR, f"contrast_{contrast_idx}", "images", filename)
            msk_path = os.path.join(DATASET_DIR, f"contrast_{contrast_idx}", "masks", filename)
            train_imgs.append(img_path)
            train_msks.append(msk_path)

    val_imgs, val_msks = [], []
    for subject_id in val_ids:
        for contrast_idx in [1, 2, 3]:
            filename = f"{subject_id:04d}.png"
            img_path = os.path.join(DATASET_DIR, f"contrast_{contrast_idx}", "images", filename)
            msk_path = os.path.join(DATASET_DIR, f"contrast_{contrast_idx}", "masks", filename)
            val_imgs.append(img_path)
            val_msks.append(msk_path)
            
    print(f"Data split by subject: {len(train_ids)} train subjects, {len(val_ids)} validation subjects.")
    print(f"Total training images: {len(train_imgs)}, Total validation images: {len(val_imgs)}")

    # Create datasets and dataloaders
    train_dataset = MRIDataset(train_imgs, train_msks)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    val_loaders = {}
    for i in [1, 2, 3]:
        c_val_imgs = [p for p in val_imgs if f"contrast_{i}" in p]
        c_val_msks = [p for p in val_msks if f"contrast_{i}" in p]
        if c_val_imgs:
            val_dataset = MRIDataset(c_val_imgs, c_val_msks)
            # UPDATED: Shuffle validation loaders for better visualization sampling
            val_loaders[f'c{i}'] = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)

    # --- Model, Loss, Optimizer ---
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    wandb.watch(model, log="all", log_freq=100)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # --- Training Loop ---
    best_val_dice = 0.0
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(); outputs = model(images); loss = loss_fn(outputs, masks)
            loss.backward(); optimizer.step(); train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_metrics_log = {}; total_val_dice = 0.0
        with torch.no_grad():
            for contrast_name, loader in val_loaders.items():
                contrast_dice_metric = 0.0; num_batches = 0
                for i, (images, masks) in enumerate(loader):
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    outputs = model(images); contrast_dice_metric += dice_metric(outputs, masks); num_batches += 1
                    if i == 0:
                        log_predictions_to_wandb(images, masks, outputs, epoch, contrast_name)
                
                avg_contrast_dice = contrast_dice_metric / num_batches if num_batches > 0 else 0
                val_metrics_log[f'dice_{contrast_name}'] = avg_contrast_dice
                total_val_dice += avg_contrast_dice

        avg_val_dice = total_val_dice / len(val_loaders) if val_loaders else 0
        val_metrics_log["dice_overall"] = avg_val_dice
        
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "validation": val_metrics_log})
        
        print(f"Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f}, Val Dice: {avg_val_dice:.4f} "
              f"(C1: {val_metrics_log.get('dice_c1', 0):.4f}, C2: {val_metrics_log.get('dice_c2', 0):.4f}, C3: {val_metrics_log.get('dice_c3', 0):.4f})")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🎉 New best model saved with Val Dice: {best_val_dice:.4f}")
            wandb.run.summary["best_val_dice"] = best_val_dice

    wandb.finish()
    print("\n🚀 Training complete!")

if __name__ == "__main__":
    main()