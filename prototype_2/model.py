# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- The UNet building blocks can be simplified and reused ---
class DoubleConv(nn.Module):
    """(convolution => => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to the size of x2
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class RegistrationUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        # The output convolution predicts the 2-channel (dx, dy) displacement field
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)

class ParameterRegressionHead(nn.Module):
    """
    A U-Net based feature extractor that regresses transformation parameters.
    """
    def __init__(self, in_channels=2, out_params=8):
        super().__init__()
        # Use the U-Net backbone to extract features
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        
        # New regression part
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Condense spatial information
            nn.Flatten(),
            nn.Linear(32, 16),       # Intermediate layer
            nn.ReLU(),
            nn.Linear(16, out_params) # Final output layer for 8 params
        )

    def forward(self, x):
        # Pass through U-Net feature extractor
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1) # Rich feature map, shape (B, 32, H, W)
        
        # Regress parameters from the feature map
        return self.regression_head(x)
    

class AttentionFusion(nn.Module):
    """
    Learns to adaptively fuse a variable number of feature maps using an
    independent, channel-wise attention mechanism (Sigmoid).
    This encourages collaborative fusion.
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        # Attention network to compute independent weights for each modality
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()  # Use Sigmoid for independent [0, 1] weights
        )

    def forward(self, feature_maps_list):
        num_contrasts = len(feature_maps_list)
        if num_contrasts == 1:
            return feature_maps_list[0]

        stacked_features = torch.stack(feature_maps_list, dim=1)
        B, N, C, H, W = stacked_features.shape

        # Reshape for parallel processing
        reshaped_features = stacked_features.view(B * N, C, H, W)
        
        # --- KEY CHANGE ---
        # Compute independent attention weights for each contrast's feature map
        attention_weights = self.attention_net(reshaped_features)
        
        # Reshape weights back to (B, N, C, 1, 1)
        attention_weights = attention_weights.view(B, N, C, 1, 1)
        
        # Apply weights and then sum the features. This is now a collaborative weighting.
        weighted_features = stacked_features * attention_weights
        fused_features = torch.sum(weighted_features, dim=1)
        
        return fused_features
    
class CrossContrastFusion(nn.Module):
    """
    Fuses a variable number of feature maps using a true cross-attention mechanism.
    This allows features from different contrasts to interact and inform each other.
    """
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, feature_maps_list):
        """
        Args:
            feature_maps_list (list of torch.Tensor): A list of N feature maps,
                                                     each of shape (B, C, H, W).
        """
        num_contrasts = len(feature_maps_list)
        if num_contrasts == 1:
            return feature_maps_list

        # Stack features and get shape info
        stacked_features = torch.stack(feature_maps_list, dim=1) # (B, N, C, H, W)
        B, N, C, H, W = stacked_features.shape

        # --- 1. Create Global Context ---
        # Average across the contrast dimension (N) to create a context summary
        # This context is permutation-invariant to the input order
        global_context = torch.mean(stacked_features, dim=1) # (B, C, H, W)

        # Prepare for attention: flatten spatial dimensions into a sequence
        # (B, C, H*W) -> (B, H*W, C)
        global_context_seq = global_context.flatten(2).permute(0, 2, 1)

        refined_features_list = []
        for i in range(num_contrasts):
            # --- 2. Cross-Attention Refinement ---
            individual_feature = stacked_features[:, i, :, :, :] # (B, C, H, W)
            individual_feature_seq = individual_feature.flatten(2).permute(0, 2, 1) # (B, H*W, C)

            # Query: from the individual contrast
            # Key/Value: from the global context of all contrasts
            # This allows each contrast to "ask" the global context for relevant info
            refined_seq, _ = self.attention(
                query=individual_feature_seq,
                key=global_context_seq,
                value=global_context_seq
            )

            # Add residual connection and layer normalization
            refined_seq = self.layer_norm(individual_feature_seq + refined_seq)

            # Reshape back to image format
            # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
            refined_feature = refined_seq.permute(0, 2, 1).view(B, C, H, W)
            refined_features_list.append(refined_feature)

        # --- 3. Final Aggregation ---
        # Sum the refined features from all contrasts
        fused_features = torch.sum(torch.stack(refined_features_list, dim=1), dim=1)

        return fused_features
    
    
class BoundaryAwareUNet(nn.Module):
    """
    A multi-task U-Net that simultaneously predicts a segmentation mask and its boundary.
    It uses a shared encoder and decoder, with two separate output heads.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # --- Encoder and Fusion modules remain the same ---
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.fuse_skip1 = CrossContrastFusion(64)
        self.fuse_skip2 = CrossContrastFusion(128)
        self.fuse_skip3 = CrossContrastFusion(256)
        self.fuse_skip4 = CrossContrastFusion(512)
        self.fuse_bottleneck = CrossContrastFusion(1024)

        # --- Decoder remains the same ---
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # --- NEW: Two separate output heads for multi-task learning ---
        self.outc_mask = OutConv(64, out_channels)
        self.outc_boundary = OutConv(64, out_channels)

    def forward(self, contrasts):
        B, N, C, H, W = contrasts.shape
        skips1, skips2, skips3, skips4, bottlenecks =[],[],[],[],[]

        for i in range(N):
            contrast_slice = contrasts[:, i, :, :, :]
            s1 = self.inc(contrast_slice)
            s2 = self.down1(s1)
            s3 = self.down2(s2)
            s4 = self.down3(s3)
            bottleneck = self.down4(s4)
            skips1.append(s1); skips2.append(s2); skips3.append(s3); skips4.append(s4); bottlenecks.append(bottleneck)

        fused_s1 = self.fuse_skip1(skips1)
        fused_s2 = self.fuse_skip2(skips2)
        fused_s3 = self.fuse_skip3(skips3)
        fused_s4 = self.fuse_skip4(skips4)
        fused_bottleneck = self.fuse_bottleneck(bottlenecks)

        x = self.up1(fused_bottleneck, fused_s4)
        x = self.up2(x, fused_s3)
        x = self.up3(x, fused_s2)
        # This is the shared feature map before the final output layers
        shared_decoder_output = self.up4(x, fused_s1)

        # --- NEW: Generate predictions from each head ---
        mask_logits = self.outc_mask(shared_decoder_output)
        boundary_logits = self.outc_boundary(shared_decoder_output)
        
        # Return a dictionary for clear, multi-task outputs
        return {"mask": mask_logits, "boundary": boundary_logits}
    
class AlignmentAndFusionUNet(nn.Module):
    """
    A wrapper model that first performs learnable, task-driven registration of
    moving contrasts to a reference contrast, and then feeds the aligned stack
    into a segmentation network.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()


        # self.registration_network = RegistrationUNet(in_channels * 2, 2)
        
        # self.registration_network.outc.conv.weight.data.zero_()
        # self.registration_network.outc.conv.bias.data.zero_()
        
        self.registration_head = ParameterRegressionHead(in_channels * 2, 8)

        # --- Initialize the final layer to output zeros ---
        # This ensures the initial deviation is zero.
        self.registration_head.regression_head[-1].weight.data.zero_()
        self.registration_head.regression_head[-1].bias.data.zero_()
        
        # --- Define the base identity matrix for a homography ---
        # A 3x3 identity matrix, flattened.
        identity_matrix = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float)
        # We only predict the first 8 params, as the 9th is fixed to 1.
        self.identity = identity_matrix[:8].view(1, 8)
        
        # --- 2. The Core Segmentation Network ---
        # We use the previously defined BoundaryAwareUNet for the main task.
        self.segmentation_network = BoundaryAwareUNet(in_channels, out_channels)

    def spatial_transformer(self, moving_image, homography_params):
        """
        Applies a perspective transformation (homography) to the moving image.
        """
        # Add the 9th element (1) to the predicted 8 params
        # This creates the full 3x3 matrix for each image in the batch
        homography_matrix = torch.cat(
            [homography_params, torch.ones(homography_params.size(0), 1, device=homography_params.device)], 
            dim=1
        ).view(-1, 3, 3)

        # PyTorch's affine_grid can't do perspective, so we build the grid manually.
        # This is a standard way to apply a homography.
        B, C, H, W = moving_image.shape
        # Create a grid of coordinates
        y_coords, x_coords = torch.meshgrid(
            [torch.linspace(-1, 1, H, device=moving_image.device), 
             torch.linspace(-1, 1, W, device=moving_image.device)],
            indexing='ij'
        )
        # Create homogeneous coordinates (x, y, 1)
        p = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1) # Shape (H, W, 3)
        p = p.view(H * W, 3).T # Shape (3, H*W)

        # Apply the inverse transformation to the grid
        # We use the inverse because we are "pulling" pixels from the source.
        p_prime = torch.inverse(homography_matrix) @ p # Shape (B, 3, H*W)

        # Convert back to inhomogeneous coordinates (divide by the new z)
        # Add a small epsilon to avoid division by zero
        p_prime = p_prime[:, :2, :] / (p_prime[:, 2, :].unsqueeze(1) + 1e-8)
        
        # Reshape to the grid format expected by grid_sample
        grid = p_prime.permute(0, 2, 1).view(B, H, W, 2) # Shape (B, H, W, 2)
        
        warped_image = F.grid_sample(moving_image, grid, align_corners=True, padding_mode="zeros")
        return warped_image


    def forward(self, reference_contrast, moving_contrasts):
        aligned_contrasts = [reference_contrast]
        predicted_deviations = []

        for moving_contrast in moving_contrasts:
            registration_input = torch.cat([reference_contrast, moving_contrast], dim=1)
            
            # Predict the *deviation* from the identity matrix
            deviation = self.registration_head(registration_input)
            predicted_deviations.append(deviation)
            
            # Add the deviation to the identity to get the final transformation
            final_params = self.identity.to(deviation.device) + deviation
            
            warped_moving_contrast = self.spatial_transformer(moving_contrast, final_params)
            aligned_contrasts.append(warped_moving_contrast)

        final_input_stack = torch.stack(aligned_contrasts, dim=1)
        seg_outputs = self.segmentation_network(final_input_stack)
        
        # Add the deviations for the regularization loss
        seg_outputs["predicted_deviations"] = torch.stack(predicted_deviations, dim=1)
        seg_outputs["warped_moving_contrasts"] = final_input_stack
        
        return seg_outputs
    
    # def spatial_transformer(self, moving_image, ddf):
    #     """
    #     Applies the predicted dense displacement field (DDF) to the moving_image,
    #     correctly handling coordinate systems.
    #     """
    #     B, C, H, W = moving_image.shape
        
    #     # 1. Create a base identity grid of pixel coordinates
    #     vectors = [torch.arange(0, s, device=ddf.device) for s in (H, W)]
    #     # 'ij' indexing gives us grids in (Height, Width) or (Y, X) order
    #     grids = torch.meshgrid(vectors, indexing='ij')
    #     y_coords = grids[0].float() # Shape (H, W)
    #     x_coords = grids[1].float() # Shape (H, W)

    #     # 2. Add the displacements from the DDF
    #     # ddf has shape (B, 2, H, W), where ddf[:, 0] is dx and ddf[:, 1] is dy
    #     dx = ddf[:, 0, ...] # Displacement in the x-direction (Width)
    #     dy = ddf[:, 1, ...] # Displacement in the y-direction (Height)
        
    #     # Add the displacement to the original coordinates
    #     # Note: x_coords and y_coords are broadcasted to the batch size B
    #     final_x_coords = x_coords + dx
    #     final_y_coords = y_coords + dy

    #     # 3. Stack the final coordinates into the (x, y) format expected by grid_sample
    #     # The final shape needs to be (B, H, W, 2)
    #     final_coords = torch.stack([final_x_coords, final_y_coords], dim=-1)

    #     # 4. Normalize the coordinates to the range [-1, 1] for grid_sample
    #     # Normalize x coordinates (dimension -1, index 0)
    #     final_coords[..., 0] = 2 * (final_coords[..., 0] / (W - 1)) - 1
    #     # Normalize y coordinates (dimension -1, index 1)
    #     final_coords[..., 1] = 2 * (final_coords[..., 1] / (H - 1)) - 1
        
    #     # 5. Sample the input image using the calculated grid
    #     warped_image = F.grid_sample(moving_image, final_coords, align_corners=True, padding_mode="zeros")
        
    #     return warped_image

    # def forward(self, reference_contrast, moving_contrasts):
    #     aligned_contrasts = [reference_contrast]
    #     # This will hold the DDFs for calculating the smoothness loss later
    #     predicted_ddfs = []

    #     for moving_contrast in moving_contrasts:
    #         registration_input = torch.cat([reference_contrast, moving_contrast], dim=1)
    #         # Predict the DDF
    #         ddf = self.registration_network(registration_input)
    #         predicted_ddfs.append(ddf)
            
    #         # Warp the moving contrast using the DDF
    #         warped_moving_contrast = self.spatial_transformer(moving_contrast, ddf)
    #         aligned_contrasts.append(warped_moving_contrast)

    #     final_input_stack = torch.stack(aligned_contrasts, dim=1)
        
    #     # Get segmentation results
    #     seg_outputs = self.segmentation_network(final_input_stack)
        
    #     # Add the DDFs and warped images to the output dict for loss calculation
    #     seg_outputs["predicted_ddfs"] = torch.stack(predicted_ddfs, dim=1)
    #     # You've already returned the warped images in your latest code, which is great
    #     # Let's adjust it to return only the *moving* ones that were warped
    #     seg_outputs["warped_moving_contrasts"] = torch.stack(aligned_contrasts, dim=1)

    #     return seg_outputs
    