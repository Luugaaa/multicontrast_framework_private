# data_loader.py

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class MultiContrastDataset(Dataset):
    """
    A PyTorch Dataset that loads multiple MRI contrasts per subject and
    applies augmentations, including simulated misalignment.
    """
    def __init__(self, subject_ids, dataset_dir, contrasts_to_use, primary_contrast_id,
                 apply_misalignment=True, 
                 max_rotation=15,
                 max_translate=0.1, 
                 max_scale=0.1):
        
        self.subject_ids = subject_ids
        self.dataset_dir = dataset_dir
        self.contrasts_to_use = contrasts_to_use
        self.primary_contrast_id = primary_contrast_id
        
        self.apply_misalignment = apply_misalignment
        self.transform_params = {
            'degrees': (-max_rotation, max_rotation),
            'translate': (max_translate, max_translate),
            'scale_ranges': [1.0 - max_scale, 1.0 + max_scale],
            'shears': None,
            'img_size': None
        }
        
        if self.primary_contrast_id not in self.contrasts_to_use:
            raise ValueError(f"Primary contrast {self.primary_contrast_id} must be in the list of contrasts to use.")

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        subject_id = self.subject_ids[index]
        
        raw_contrast_images = []
        for contrast_idx in self.contrasts_to_use:
            filename = f"{subject_id:04d}.png"
            img_path = os.path.join(self.dataset_dir, f"contrast_{contrast_idx}", "images", filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.transform_params['img_size'] = image.shape[:2]
            raw_contrast_images.append(T.ToTensor()(image))
        
        raw_masks = []
        for contrast_idx in self.contrasts_to_use:
            mask_filename = f"{subject_id:04d}.png"
            mask_path = os.path.join(self.dataset_dir, f"contrast_{contrast_idx}", "masks", mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            raw_mask = T.ToTensor()(mask)
            raw_masks.append(raw_mask)

        final_contrasts = [None] * len(self.contrasts_to_use)
        final_masks = [None] * len(self.contrasts_to_use)
        primary_idx_in_list = self.contrasts_to_use.index(self.primary_contrast_id)

        if self.apply_misalignment:
            # transform_params = T.RandomAffine.get_params(**self.transform_params)
            
            # primary_img_tensor = raw_contrast_images[primary_idx_in_list]
            # final_contrasts[primary_idx_in_list] = T.functional.affine(primary_img_tensor, *transform_params)
            # final_mask = T.functional.affine(raw_mask, *transform_params)

            for i, (raw_img, raw_mask) in enumerate(zip(raw_contrast_images, raw_masks)):
                # if i == primary_idx_in_list:
                #     continue
                
                other_transform_params = T.RandomAffine.get_params(**self.transform_params)
                final_contrasts[i] = T.functional.affine(raw_img, *other_transform_params)
                final_masks[i] = T.functional.affine(raw_mask, *other_transform_params)
        else:
            final_contrasts = raw_contrast_images
            final_masks = raw_masks
            
        contrasts_tensor = torch.stack(final_contrasts)
        masks_tensor = torch.stack(final_masks)
        
        return {
            'subject_id': subject_id,
            'contrasts': contrasts_tensor,
            'masks': masks_tensor,
        }