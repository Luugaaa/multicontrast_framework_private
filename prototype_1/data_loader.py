import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- 1. Configuration ---
# You can change these settings for any training run
DATASET_DIR = "datasets/mri_dataset"
CONTRASTS_TO_USE = [1] # Use all 4 contrasts for this example
PRIMARY_CONTRAST_IDX = 1       # The segmentation will be generated on Contrast 1
NUM_SUBJECTS = 200
BATCH_SIZE = 8
VAL_SPLIT = 0.2

# --- 2. The Multi-Contrast Dataset Class ---

class MultiContrastDataset(Dataset):
    """
    A PyTorch Dataset that loads multiple contrast images for a single subject.
    """
    def __init__(self, subject_ids, dataset_dir, contrasts_to_use, primary_contrast_idx=1):
        """
        Args:
            subject_ids (list): List of subject IDs for this dataset split (e.g., train or val).
            dataset_dir (str): Path to the root dataset directory.
            contrasts_to_use (list): List of integers specifying which contrasts to load.
            primary_contrast_idx (int): The index of the main contrast for segmentation.
        """
        self.subject_ids = subject_ids
        self.dataset_dir = dataset_dir
        self.contrasts_to_use = sorted(contrasts_to_use) # Keep a consistent order
        self.primary_contrast_idx = primary_contrast_idx
        
        if self.primary_contrast_idx not in self.contrasts_to_use:
            raise ValueError(f"Primary contrast {self.primary_contrast_idx} must be in the list of contrasts to use.")

    def __len__(self):
        """Returns the number of subjects in the dataset."""
        return len(self.subject_ids)

    def __getitem__(self, index):
        """
        Fetches all specified contrasts and the mask for a single subject.
        """
        subject_id = self.subject_ids[index]
        
        contrast_images = []
        for contrast_idx in self.contrasts_to_use:
            filename = f"{subject_id:04d}.png"
            img_path = os.path.join(self.dataset_dir, f"contrast_{contrast_idx}", "images", filename)
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Normalize to [0, 1] and add channel dimension
            image_tensor = torch.from_numpy((image.astype(np.float32) / 255.0)[np.newaxis, :, :])
            contrast_images.append(image_tensor)
            
        # Stack contrasts into a single tensor: [num_contrasts, 1, H, W]
        contrasts_tensor = torch.stack(contrast_images)

        # Load the mask (it's the same for all contrasts of a subject)
        mask_filename = f"{subject_id:04d}.png"
        mask_path = os.path.join(self.dataset_dir, f"contrast_1", "masks", mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_tensor = torch.from_numpy((mask.astype(np.float32) / 255.0)[np.newaxis, :, :])
        
        # Find the index of the primary contrast within our loaded list
        # This tells the decoder which feature map to focus on.
        primary_idx_in_batch = self.contrasts_to_use.index(self.primary_contrast_idx)

        return {
            'subject_id': subject_id,
            'contrasts': contrasts_tensor,
            'mask': mask_tensor,
            'primary_idx': primary_idx_in_batch
        }

# --- 3. Demonstration ---

if __name__ == "__main__":
    # 1. Split subject IDs into train and validation sets
    all_subject_ids = list(range(NUM_SUBJECTS))
    train_ids, val_ids = train_test_split(all_subject_ids, test_size=VAL_SPLIT, random_state=42)
    
    print(f"✅ Using contrasts: {CONTRASTS_TO_USE}")
    print(f"✅ Primary contrast for segmentation is: Contrast {PRIMARY_CONTRAST_IDX}")
    print("-" * 30)

    # 2. Create an instance of the dataset for the training subjects
    train_dataset = MultiContrastDataset(
        subject_ids=train_ids,
        dataset_dir=DATASET_DIR,
        contrasts_to_use=CONTRASTS_TO_USE,
        primary_contrast_idx=PRIMARY_CONTRAST_IDX
    )
    
    # 3. Create a DataLoader
    # The default collate function will automatically stack the dictionaries into a batch
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 4. Fetch one batch to inspect its structure
    print("Fetching one batch from the DataLoader...")
    first_batch = next(iter(train_loader))
    
    print("\n--- Batch Structure ---")
    print(f"Batch keys: {first_batch.keys()}")
    
    contrasts_batch = first_batch['contrasts']
    mask_batch = first_batch['mask']
    primary_idx_batch = first_batch['primary_idx']
    
    print(f"\nShape of 'contrasts' tensor: {contrasts_batch.shape}")
    print(" -> (Batch Size, Num Contrasts, Channels, Height, Width)")
    
    print(f"Shape of 'mask' tensor: {mask_batch.shape}")
    print(" -> (Batch Size, Channels, Height, Width)")
    
    print(f"Primary contrast indices in batch: {primary_idx_batch}")
    print(f" -> (Tells us that for every item, the image at index {primary_idx_batch[0]} is the primary one)")